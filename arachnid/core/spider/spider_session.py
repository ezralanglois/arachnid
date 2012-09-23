'''
This defines a bridge between Python and Spider. 

Adapted from http://www.wadsworth.org/spider_doc/spider/proc/spyder.py

Example usage
-------------
    
.. sourcecode:: py

    >>> from spider.core.spider_session import Session
    >>> spi = Session(ext='dat')
    >>> spi.invoke("[size]=117")
    >>> size = spi['size']
    >>> print "---------------------- size =  %f" % size
    >>> spi.invoke("x11=7.7")
    >>> x11 = spi['x11']
    >>> print "---------------------- x11 =  %f" % x11
    >>> spi['x12'] = 7.7
    >>> print "---------------------- x11 =  %f" % spi['x12']
    >>> spi.set(x13=1.0, size=12)
    >>> print "---------------------- x11 =  %f, size = %d" % spi['x13'], spi['size']
    >>> spi.invoke("MO", "tmp001", "64,64", "T")
    >>> spi.close()


. todo::

    add warnings based on Spider Version

.. Created on Aug 13, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import sys, re, struct, os, logging, subprocess, tempfile, glob, select as io_select, atexit, math
import spider_var
from spider_parameter import spider_image, spider_tuple, spider_stack, is_incore_filename
import spider_parameter
#from collections import defaultdict

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class Session(object):
    ''' Create an interactive session with Spider
    
    :Parameters:
        
        filepath : str
                   Filename for the spider executable (Default: None) if None, a search is performed
        ext : str
              File extension for data files (Default: dat)
        thread_count : int
                       Number of threads to use (Default: 0) if 0, all cores are used
        enable_results : bool
                         Should the results file be enabled for the entire run
        rank : int
               Current rank if MPI job
        tmp_path : str
                   Current working directory of SPIDER
    '''
    
    EXTERNAL_PIPENAME = 'TMP_SPIDER_PIPE.pipe'
    
    PROCESSES=[]
    
    def __init__(self, filepath=None, ext='dat', thread_count=0, enable_results=False, rank=None, tmp_path=None):
        #Invoke Spider and open a pipe
        
        self.spider = None
        self.registers = None
        self.dataext = ext if ext[0] != '.' else ext[1:]
        self.version = None
        self._results = True
        
        if tmp_path is not None and tmp_path != "" and not os.path.exists(tmp_path):
            _logger.warn("Local path (--local-temp) does not exist: %s"%tmp_path)
            tmp_path=None
        
        if rank is not None: _logger.debug("Using MPI spider = %d"%rank)
        if tmp_path: _logger.debug("Using local path: %s"%tmp_path)
        if filepath == None or filepath == "":
            _logger.debug("Searching for Spider")
            spiderexec = 'spider_linux_mp_opt64'
            if os.environ.has_key('SPIDER_ROOT') and os.environ['SPIDER_ROOT'] != "":
                self.spiderexec = os.path.join(os.environ['SPIDER_ROOT'], 'bin', spiderexec)
                _logger.debug(" -- Found @ SPIDER_ROOT: %s"%self.spiderexec)
            elif os.environ.has_key('SPIDER_LOC') and os.environ['SPIDER_LOC'] != "":
                self.spiderexec = os.path.join(os.environ['SPIDER_LOC'], spiderexec)
                _logger.debug(" -- Found @ SPIDER_LOC: %s"%self.spiderexec)
            else:
                if os.sys.platform.startswith('darwin'):
                    self.spiderexec = "/Applications/spider/bin/spider_osx_64"
                else:
                    self.spiderexec = '/guam.raid.cluster.software/spider.19.11/spider_linux_mp_opt64.19.11' #'/guam.raid.cluster.software/bin/spider_linux_mp_opt64'
                    if not os.path.exists(self.spiderexec):
                        #self.spiderexec = '/home/ezra/spider/bin/spider_linux_mp_intel64'
                        self.spiderexec = '/home/ezra/spider/bin/spider_linux_mp_opt64'
                _logger.debug(" -- Found default: %s"%self.spiderexec)
        else:
            self.spiderexec = filepath
            _logger.debug("Using spider: %s"%self.spiderexec)
        if not os.path.exists(self.spiderexec): raise ValueError, "Cannot find spider executable: "+self.spiderexec
        self.get_version(tmp_path)
        if rank == 0:
            _logger.info("SPIDER Version = %d.%d"%self.version)
        
        if os.path.exists(os.path.join(os.path.dirname(self.spiderexec), 'Nextresults')):
            os.environ['SPBIN_DIR'] = os.path.dirname(self.spiderexec) + os.sep
            _logger.debug("SPBIN_DIR = %s"%os.environ['SPBIN_DIR'])
        else:
            os.environ['SPBIN_DIR'] = '/guam.raid.cluster.software/spider.18.15/bin/'
        
        self.pipename = Session.EXTERNAL_PIPENAME #os.path.abspath(
        if rank is not None:
            base, ext = os.path.splitext(self.pipename)
            self.pipename = base+("_%d"%rank)+ext
        else: rank=0
        if tmp_path is not None and tmp_path != "": 
            self.pipename = os.path.join(tmp_path, self.pipename)
        if os.path.exists(self.pipename):
            try: os.remove(self.pipename)
            except: pass
        _logger.debug("Using PIPE = %s"%self.pipename)
        os.mkfifo(self.pipename)
        if tmp_path == "": tmp_path = None
        if 1 == 0:
            self.spider_err = tempfile.NamedTemporaryFile(prefix='SPIDER_ERR_%d'%rank, dir=tmp_path, delete=True)
            self.spider = subprocess.Popen(self.spiderexec, cwd=tmp_path, stdin=subprocess.PIPE, stderr=self.spider_err.fileno()) #stdout=subprocess.PIPE
        else:
            self.spider = subprocess.Popen(self.spiderexec, cwd=tmp_path, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            self.spider_err = self.spider.stderr
        Session.PROCESSES.append(self.spider)
        self.spider_poll = io_select.poll()
        self.spider_poll.register(self.spider_err.fileno())
        self._invoke(self.dataext)
        if _logger.getEffectiveLevel() == logging.DEBUG and enable_results: 
            self._invoke('MD', 'RESULTS ON')
            self._invoke('MD', 'TERM ON') 
            _logger.debug("Result enabled for terminal")
        else: 
            self._invoke('MD', 'RESULTS OFF')
            self._invoke('MD', 'TERM OFF') 
            self._results = False
        self._invoke('MD', 'PIPE', self.pipename)
        self._invoke('MD', 'SET MP', thread_count)
        
        self.registers = open(self.pipename, 'r')
        self.register_poll = io_select.poll()
        self.register_poll.register(self.registers.fileno())
        self.incore_imgs = spider_var.spider_var_pool()
        self.incore_docs = spider_var.spider_var_pool()
        
        if is_linux():
            try:
                self._invoke("[i] = 3.241","PI REG", "[i]") #, skip=True)
            except: pass
            response = ''
            while len(response) < 9:
                response += self.registers.readline()
            struct.unpack('ffc',response)
    
    def get_version(self, tmp_path=None):
        ''' Get the current version of SPIDER
        
        :Parameters:
        
        tmp_path : str, optional
                   Current working directory for SPIDER
        
        :Returns:
        
        version : tuple of ints
                  Version tuple 
        '''
        
        if self.version is None:
            if tmp_path == "": tmp_path = None
            spider_version = subprocess.Popen(self.spiderexec, cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            spider_version.stdin.write('tmp\n')
            spider_version.stdin.write('en d\n')
            spider_version.stdin.flush()
            n=-1
            while True:
                line = spider_version.stdout.readline()
                if line == "": break
                n = line.find('VERSION:')
                if n != -1: break
            if n == -1:
                raise ValueError, "Error invoking SPIDER executable on path: %s - This could mean your tmp_path flag is not write accessible"%self.spiderexec
            line = line[(n+len('VERSION:')+1):]
            line = line.strip().split()
            self.version = tuple([int(v) for v in line[1].split('.')])
            spider_version.stderr.close()
            spider_version.stdout.close()
            spider_version.stdin.close()
        return self.version
    
    def temp_incore_image(self, force_new=False, hook=None, is_stack=False):
        ''' Create a unique incore file integer
        
        :Parameters:
            
        force_new : bool
                    If true, ensure the new variable as never been used
        hook : function
               Called when the variable is deleted
        is_stack : bool
                   True if temp file is a stack
        
        :Returns:
            
        var : int
              Integer referring to an incore spider variable
        '''
        
        return self.incore_imgs.get(force_new, hook, is_stack)
    
    def temp_incore_doc(self, force_new=False, hook=None):
        ''' Create a unique incore file integer
        
        :Parameters:
            
        force_new : bool
                    If true, ensure the new variable as never been used
        hook : function
               Called when the variable is deleted
        
        :Returns:
            
        var : int
              Integer referring to an incore spider variable
        '''
        
        return self.incore_docs.get(force_new, hook)
    
    def ext(self):
        '''Get the current data extension for spider
        
        :Returns:
            
        ext : str
              Current data extension for images and doc files
        '''
        
        return self.dataext
    
    def replace_ext(self, filename):
        '''Replace the extension of the given filename with the appropriate spider extension
        
        :Parameters:
            
        filename : str
                   Filename with possibly offending extension
                       
        :Returns:
            
        filename : str
                   Filename with proper SPIDER extension
        '''
        
        if is_incore_filename(filename) or isinstance(filename, tuple): return filename
        
        try:
            idx = filename.find('@')
            if idx >= 0: filename = filename[:idx]
        except: pass
        
        filename, tmp_ext = os.path.splitext(filename)
        return filename + "." + self.dataext
    
    def __del__(self):
        ''' Close the Spider session nicely
        '''
        
        self.close()
    
    def close(self):
        ''' Close the Spider session
        '''
        
        _logger.debug("Attempting to close SPIDER")
        if hasattr(self, 'spider') and self.spider is not None:
            _logger.debug("Closing SPIDER")
            if _logger.getEffectiveLevel() > logging.DEBUG:
                self.spider.stdin.write('en d\n')
            else:
                self.spider.stdin.write('en d\n')
            self.spider.stdin.flush()
            self.spider = None
        
        if hasattr(self, 'registers') and self.registers is not None:
            self.registers.close()
            self.registers = None
        
        if hasattr(self, 'pipename') and self.pipename is not None:
            try: os.remove(self.pipename)
            except: pass
        tmp_files = ['jnkASSIGN1', 'LOG.%s'%self.dataext, 'LOG.tmp']
        if not self._results:
            tmp_files.extend(glob.glob('results.%s.*'%self.dataext))
        tmp_files.extend(glob.glob('fort.*'))
        tmp_files.extend(glob.glob('_*.'+self.dataext))
        tmp_files.extend(glob.glob('_*.%s'%self.dataext))
        if _logger.getEffectiveLevel() > logging.DEBUG:
            tmp_files.extend(glob.glob('results.%s.*'%self.dataext))
        for filename in tmp_files:
            if os.path.exists(filename):
                try: os.remove(filename)
                except: pass
    
    def _invoke(self, *args, **kwargs):
        ''' Send a command to Spider
        
        :Parameters:
            
        args : list
               List of arguments
        kwargs : dict
                 Keyword arguments
        '''
        
        if self.spider is None: raise ValueError, "No pipe to Spider process"
        test_error = kwargs.get('test_error', True)
        if not test_error:_logger.error("Error results: %s"%str(args))
        for arg in args:
            cmd = str(arg)
            self.spider.stdin.write(cmd+'\n')
            try:
                err = self._get_errors()
            except: pass
            else:
                if err is not None:
                    if test_error and 1 == 0:
                        try:
                            self._invoke_with_results(*args) 
                        except: pass
                    _logger.error("Error in command: %s - with message: %s"%(str(args), str(err)))
                    if test_error: 
                        raise SpiderCommandError, err
        self.spider.stdin.flush()
        
    def _get_errors(self):
        #A private function used by invoke() to get the current error file
        
        line = None
        if 1 == 1 and len(self.spider_poll.poll(10)):
            line = ""
            line += self.spider_err.read(1)
            while len(self.spider_poll.poll(10)):
                line += self.spider_err.read(1)
        if line is not None and line.find('Warning') != -1: line = None
        return line
        
    def invoke(self, *args):
        ''' Send a command to Spider
        
        If an error occurs, then this function tries to turn on the results file
        and run the command again.
        
        :Parameters:
            
        args : list
               List of arguments
        
        '''
        
        _logger.debug("%s"%str(args))
        self._invoke(*args)
        if self[9] > 0:
            _logger.error("Error in command: %s"%str(args))
            for arg in args:
                _logger.error("Arg: %s"%arg)
            self._invoke_with_results(*args) 
            self[9] = "0"
            _logger.error("Error in command: %s"%str(args))
            raise SpiderCommandError, "%s failed in Spider"%args[0]
    
    def _invoke_with_results(self, *args):
        ''' Invoke a SPIDER command with the results file on
        
        :Parameters:
            
        args : list
               List of arguments
        '''
        
        if not self._results:
            self._invoke('MD', 'RESULTS ON', test_error=False)
            _logger.error("here: %f"%self[9])
            self._invoke('MD', 'TERM ON', test_error=False) 
            self._invoke(*args, test_error=False)
            self._invoke('my fl', test_error=False)
            self._invoke('MD', 'RESULTS OFF', test_error=False)
            self._invoke('MD', 'TERM OFF', test_error=False) 
        
    
    def set(self, **kwargs):
        ''' Set the spider register with given name and value
        
        :Parameters:
            
        kwargs : dict
                 (Name,Value) pairs of register
        '''
        
        if self.registers is None: raise ValueError, "No pipe from Spider process"
        for varname, value in kwargs.iteritems():
            varname = spider_register_name(varname)
            self._invoke("%s=%s"%(varname, str(value)))
    
    def get(self, varname):
        ''' Get the value of the given variable name
        
        :Parameters:
            
        varname : str
                  Name of variable
        
        :Returns:
            
        val : object
              Value of variable
        '''
        
        if self.registers is None: raise ValueError, "No pipe from Spider process"
        varname = spider_register_name(varname)
        self._invoke('PI REG', varname)
        res = ''
        #self.register_poll
        while self.spider_poll.poll(10) or len(res) < 13:
            if self.spider.poll(): raise StandardError, "SPIDER has terminated"
            res += self.registers.readline()
        
        #while len(res) < 13:
        #    try:
        #        res += self.registers.readline()
        #    except: continue
        return unpack_register(res)
    
    def __getitem__(self, varname):
        ''' Get the value of the given variable name
        
        :Parameters:
            
        varname : str
                  Name of variable
        
        :Returns:
            
        val : object
              Value of variable
        '''
        
        return self.get(varname)
    
    def __setitem__(self, varname, value):
        ''' Set the value of the given variable name
        
        :Parameters:
            
        varname : str
                  Name of variable
        value : object
                Value of variable
        '''
        
        self.set(**{spider_register_name(varname): value})


def spider_command_fifo(session, command, inputfile, outputfile, message, *args):
    '''Template function for a spider command that reads a single input file
    and writes a single outputfile
    
    :Parameters:
        
        session : Session
                  Current spider session
        command : str
                  Command name
        inputfile : str
                    Filename of input image
        outputfile : str
                     Filename of the output image (If None, create an incore file)
        message : str
                  Message to be written to the log file
        args : list
                List of positional arguments to pass on
    
    :Returns:
            
        outputfile : str
                     Filename of output
    '''
    
    _logger.debug(message)
    if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
    session.invoke(command, spider_image(inputfile), spider_image(outputfile), *args)
    return outputfile

def spider_command_multi_input(session, command, message, inputfile, *otherfiles, **extra):
    '''Template function for a spider command that works on a varying number of input files
    
    :Parameters:
        
    session : Session
              Current spider session
    command : str
              Command name
    message : str
              Message to be written to the log file
    inputfile : str
                Input filename
    otherfiles : str
                 Other filenames listed on the as function parameters
    extra : dict
            Unused key word arguments (outputfile hidden here and used)
    
    :Returns:
    
    outputfile : str
                 Output filename
    '''
    
    _logger.debug(message)
    outputfile = extra.get('outputfile', None)
    if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
    if len(otherfiles) == 1 and otherfiles[0] is None: return session.cp(inputfile)
    otherfiles = [spider_image(f) for f in otherfiles]
    otherfiles.extend([spider_image(outputfile), '*'])
    session.invoke(command, spider_image(inputfile), *otherfiles)
    return outputfile

def generate_ctf_param(defocus, cs=None, window=None, source=None, defocus_spread=None, ampcont=None, envelope_half_width=None, astigmatism=0.0, azimuth=0.0, maximum_spatial_freq=None, apix=None, elambda=None, voltage=None, pad=None, ctf_sign=-1, **extra):
    ''' Generate CTF parameters for SPIDER CTF functions
    
    :Parameters:
    
    defocus : float
              Amount of defocus, in Angstroems
    cs : object
         Spherical aberration constant
    window : int
             Dimension of the 2D array
    source : float
             Size of the illumination source in reciprocal Angstroems
    defocus_spread : float
                     Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
    ampcont : float
              Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
    envelope_half_width : float
                          Envelope parameter specifies the 2 sigma level of the Gaussian
    astigmatism : float
                  Defocus difference due to axial astigmatism (Defaut: 0)
    azimuth : float
              Angle, in degrees, that characterizes the direction of astigmatism (Defaut: 0)
    maximum_spatial_freq : float
                           Spatial frequency radius corresponding to the maximum radius (Defaut: None)
    pad : int
          Number of times to pad image and CTF
    apix : float
           Size of pixel in angstroms  (Defaut: None)
    elambda : float
              Wavelength of the electrons (Defaut: None)
    voltage : float
              Voltage of microscope (Defaut: None)
    ctf_sign : float
               Application of the transfer function results in contrast reversal if underfocus (Defaut: -1)
    extra : dict
            Unused key word arguments
            
    :Returns:
    
    param : tuple
            Properly formated parameters
    '''
    
    if isinstance(cs, dict):
        extra.update(cs)
        return generate_ctf_param(defocus, **extra)
    if elambda is None: 
        if voltage is None: raise spider_parameter.SpiderParameterError, "Wavelength of the electrons is not set as elambda or voltage"
        elambda = 12.398 / math.sqrt(voltage * (1022.0 + voltage))
    if maximum_spatial_freq is None: 
        if apix is None: raise spider_parameter.SpiderParameterError, "patial frequency radius corresponding to the maximum radius is not set as maximum_spatial_freq or apix"
        maximum_spatial_freq = 0.5/apix
    if pad is None or pad < 1: pad = 1
    if isinstance(window, tuple): window = (window[0]*pad, window[1]*pad)
    else: window = (window*pad, )
    return spider_tuple(cs), spider_tuple(defocus, elambda), spider_tuple(*window), spider_tuple(maximum_spatial_freq), spider_tuple(source, defocus_spread), spider_tuple(astigmatism, azimuth), spider_tuple(ampcont, envelope_half_width), spider_tuple(ctf_sign)

def ensure_output_recon3(session, outputfile):
    ''' Ensure correct output filename names for the
    SPIDER reconstruction engine, e.g. bp 32f
    
    :Parameters:
        
    session : Session
              Current spider session
    outputfile : tuple
                 Tuple of 3 filenames
        
    :Returns:
    
    outputfile : str
                 Output full volume
    outputfile1 : str
                  Output half volume
    outputfile2 : str
                  Output half volume
    '''
    
    if outputfile is None: 
        outputfile = (session.temp_incore_image(hook=session.de), session.temp_incore_image(hook=session.de), session.temp_incore_image(hook=session.de))
    elif isinstance(outputfile, str):
        outputfile = (outputfile, prefix(outputfile, 'h1'), prefix(outputfile, 'h2'))
    elif not isinstance(outputfile, tuple) or len(outputfile) != 3:
        raise ValueError, "outputfile must be None (default), a string or a tuple of 3 strings"
    else:
        if outputfile[0] is None: outputfile = (session.temp_incore_image(hook=session.de), outputfile[1], outputfile[2])
        if outputfile[1] is None: outputfile = (outputfile[0], session.temp_incore_image(hook=session.de), outputfile[2])
        if outputfile[2] is None: outputfile = (outputfile[0], outputfile[1], session.temp_incore_image(hook=session.de))
    return outputfile
    
def ensure_stack_select(session, stack, select=None):
    ''' Ensure a proper selection file and maximum stack count
    
    :Parameters:
    
    session : Session
              Current spider session
    stack : str
            Name of the stack
    select : str, optional
             Name of the selection file
    
    :Returns:
    
    select : str
             New selection file (if select None, empty or '*')
    stack_count : int
                  Maximum offset in stack
    stack_size : int
                 Number of slices in the stack
    '''
    
    if select is None or select == "" or select == "*":
        if stack is None: raise ValueError, "Requires stack filename"
        stack_count, = session.fi_h(spider_stack(stack), ('MAXIM'))
        select = (1, int(stack_count))
    stack_count, stack_size, stack_size = session.ud_n(select)
    return select, int(stack_count), int(stack_size)

_is_spider_regiser = re.compile("[xX]\d\d")

def is_linux():
    ''' Test if the current OS is linux
    
    :Returns:
        
    val : bool
          True if platform is linux
    '''
    
    return sys.platform.startswith('linux')

def spider_register_name(varname):
    ''' Get the name of a spider register
        
    :Parameters:
            
    varname : str
              Name of variable
        
    :Returns:
            
    val : str
          Name of Spider register
    '''
    
    if isinstance(varname, int):
        return "[_%d]"%varname
    varname = varname.strip()
    if varname[0] != '[' and not is_spider_register(varname): #[9]
        varname = '[' + varname + ']'
    return varname
    
def is_spider_register(val):
    ''' Test if string value holds spider register
    
    :Parameters:
        
    val : str
          String possibly containing spider register
    
    :Returns:
        
    val : bool
          True if it contains a spider register
    '''
    
    return _is_spider_regiser.match(val)

def prefix(filename, tag):
    '''Prefix a filename with the given tag
    
    :Parameters:

    filename : str
               File path name
    tag : str
          Prefix for name of the file
    
    :Returns:
    
    out : str
         File path name where name of the file has a prefix
    '''
    
    path, name = os.path.dirname(filename), os.path.basename(filename)
    return os.path.join(path, tag+"_"+name)

def unpack_register_linux(data):
    ''' Unpack values from a register
    
    :Paramters:
        
    data : str
           Message from spider program
    
    :Returns:
    
    val : tuple
          Registry value
    '''
    
    return struct.unpack('fffc',data)[2]

def unpack_register_other(data):
    ''' Unpack values from a register
    
    :Paramters:
        
    data : str
           Message from spider program
    
    :Returns:
    
    val : tuple
          Registry value
    '''
    
    return struct.unpack('fc',data)[0]

if is_linux(): unpack_register = unpack_register_linux
else: unpack_register = unpack_register_other

class SpiderCommandError(StandardError): 
    ''' Exception is raised when SPIDER reports an error after
    a command is invoked.
    '''
    pass

def cleanup():
    ''' Clean up any living SPIDER processes
    '''
    
    for proc in Session.PROCESSES:
        proc.kill()

atexit.register(cleanup)

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    spi = Session(ext='dat')
    spi.invoke("[size]=117")
    size = spi['size']
    print "---------------------- size =  %f" % size
    spi.invoke("x11=7.7")
    x11 = spi['x11']
    print "---------------------- x11 =  %f" % x11
    spi['x12'] = 7.7
    print "---------------------- x11 =  %f" % spi['x12']
    spi.set(x13=1.0, size=12)
    print "---------------------- x11 =  %f, size = %d" % spi['x13'], spi['size']
    spi.close()
    
    