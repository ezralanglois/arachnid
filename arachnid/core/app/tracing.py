''' Common logging utilities and setup for all scripts

Every script supports logging various states of progress and this can be controlled
with a common interface.

Parameters
-----------

The tracing script has the following inheritable parameters:

.. option:: -v <CHOICE>, --log-level <CHOICE>
    
    Set logging level application wide: 'critical', 'error', 'warning', 'info', 'debug' or 0-4

.. option:: --log-file <FILENAME>
    
    Set file to log messages

.. option:: --log-config <FILENAME>
    
    File containing the configuration of the application logging

.. option:: --disable_stderr <BOOL>
    
    If true, output will only be written to the given log file

Examples
--------

.. sourcecode:: sh
    
    $ ap-bench -v4                   # Log everything including debug information
    $ ap-bench --log-level 0         # Do not log any messages
    $ ap-bench --log-level info      # Log only information level or higher
    
    $ ap-bench --log-file info.log   # Write all log messages to a file 

The logging framework also supports fine-grained configuration. This can be enabled using
the `--log-config` parameter.

The following is an example configuration file. Logging can be configured for a specific 
module or an entire package. 

::

    [loggers]
    keys=root,arachnid
    
    [handlers]
    keys=consoleHandler
    
    [formatters]
    keys=simpleFormatter
    
    [logger_root]
    level=DEBUG
    handlers=consoleHandler
    
    [logger_arachnid]
    level=DEBUG
    handlers=consoleHandler
    qualname=arachnid
    
    [handler_consoleHandler]
    class=StreamHandler
    level=DEBUG
    formatter=simpleFormatter
    args=(sys.stdout,)
    
    [formatter_simpleFormatter]
    format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
    datefmt=

The above logging configuration file can be specified on the command-line.

.. sourcecode:: sh
    
    $ ap-bench --log-config log.conf # Configure logging with a file

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import logging, logging.config, os, socket, time, sys, zipfile

loaded = False

log_level_val = ['critical', 'error', 'warning', 'info', 'debug']
log_level_map = {'critical': logging.CRITICAL,
                 'error':    logging.ERROR,
                 'warning':  logging.WARNING,
                 'info':     logging.INFO,
                 'debug':    logging.DEBUG, }
log_formats = { 'critical': "%(asctime)s %(message)s",
                'error':    "%(asctime)s %(levelname)s %(message)s",
                'warning':  "%(asctime)s %(levelname)s %(message)s",
                'info':     "%(asctime)s %(levelname)s %(message)s",
                'debug':    "%(asctime)s:%(lineno)d:%(name)s:%(levelname)s - %(message)s" }

def log_import_error(message, logger=None):
    ''' Create a logger with a stream handler and log a warning in
     non-debug mode and and exception in debug mode
     
    :Parameter:
     
    message : str
               Message to log
    logger : Logger, optional
             Specific logger to use
    '''
    
    if logger is None: logger = logging.getLogger()
    num_handlers=len(logger.handlers)
    if num_handlers == 0: logger.addHandler(logging.StreamHandler())
    if logger.isEnabledFor(logging.DEBUG): logger.exception(message)
    else: logger.warn(message)
    if num_handlers == 0: logger.removeHandler(logger.handlers[0])
    
def setup_options(parser, pgroup=None):
    '''Add options to the given option parser
    
    :Parameters:
    
    parser : optparse.OptionParser
           Program option parser
    pgroup : optparse.OptionGroup
            Parent option group
    '''
    from settings import OptionGroup
    levels=tuple(log_level_val)
    group = OptionGroup(parser, "Logging", "Options to control the state of the logging module", id=__name__)
    if pgroup is None: pgroup = group
    group.add_option("-v", log_level=levels,    help="Set logging level application wide", default=3)
    group.add_option("",   log_file="",         help="Set file to log messages", gui=dict(filetype="save"), archive=True)
    group.add_option("",   log_config="",       help="File containing the configuration of the application logging", gui=dict(filetype="open"))
    group.add_option("",   disable_stderr=False, help="If true, output will only be written to the given log file")
    parser.add_option_group(group)

def configure_logging(rank=0, log_level=3, log_file="", log_config="", remote_tmp="", disable_stderr=False, **extra):
    '''Configure logging with use selected options

    .. sourcecode:: py
    
        >>> import core.app.tracing, logging
        >>> core.app.tracing.configure_logging(log_level=3)
    
    :Parameters:

    log_level : int
                Level for logging application wide
    log_file : str
               File path for logging messages
    log_config : str
                 File path for logging configuration
    disable_stderr : bool
                     Do not redirect to stderr
    extra : dict
            Unused keyword arguments
    '''
    
    if log_config != "":
        logging.config.fileConfig(log_config)
    else:
        if rank != 0 and log_file != "":
            if log_file != "":
                base, ext = os.path.splitext(log_file)
                base += "_"
            else: base, ext = "", ".log"
            log_file = base+socket.gethostname()+"_"+str(rank)+ext
            #if remote_tmp != "": log_file = os.path.join(remote_tmp, log_file)
        handlers = []
        default_error_log = "."+os.path.basename(sys.argv[0])+".crash_report"
        
        try: 
            if log_file != "":
                if not disable_stderr: 
                    h = logging.StreamHandler()
                    h.addFilter(ExceptionFilter())
                    handlers.append(h)
                logging.debug("Writing to log file: %s"%(log_file))
                backupname = backup(log_file)
                if backupname: logging.warn("Backing up log file to %s"%(backupname))
                h = logging.FileHandler(log_file, mode='w')
                h.addFilter(ExceptionFilter())
                handlers.append(logging.FileHandler(default_error_log, mode='w'))
                handlers.append(h)
            else:
                h = logging.StreamHandler()
                h.addFilter(ExceptionFilter())
                handlers.append(logging.FileHandler(default_error_log, mode='w'))
                handlers.append(h)
        except: 
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.exception("Logging to %s"%log_file)
            ch = logging.StreamHandler()
            ch.addFilter(ExceptionFilter())
            handlers.append(logging.FileHandler(default_error_log, mode='w'))
            handlers.append(ch)
        try:    log_level = log_level_val[log_level]
        except: pass
        level = log_level_map[log_level]
        logging.basicConfig(level=level)
        root = logging.getLogger()
        root.setLevel(level)
        while len(root.handlers) > 0: root.removeHandler(root.handlers[0])
        for ch in handlers: 
            ch.setFormatter(logging.Formatter(log_formats[log_level]))
            root.addHandler(ch)
            ch.setLevel(level)
        root.setLevel(level)
'''   
def archive(parser, archives, archive_path, config_file, **extra):
    
    if archive_path != "":
        progname = os.path.basename(sys.argv[0])
        if progname.find('.') != -1: progname = os.path.splitext(progname)[0]
        progname += datetime.now().strftime("%Y%m%d_%H%M%S")+".zip"
        confname = ".tmp."+os.path.basename(config_file)
        parser.write(confname)
        zf = zipfile.ZipFile(os.path.join(archive_path, progname), mode='w')
        try:
            for filename in archives:
                if filename == "": continue
                _logger.debug("Archived %s"%filename)
                zf.write(filename, arcname=os.path.basename(filename), compress_type=zipfile.ZIP_STORED)
            zf.write(confname, arcname=os.path.basename(confname), compress_type=zipfile.ZIP_STORED)
        finally:
            zf.close()
        os.unlink(confname)
'''

def backup(filename):
    ''' Save existing file in a zip archive of the same name but '.zip' extension. The file
    will be given a unique name based on the current date and time.
    
    :Parameters:
    
    filename : str
               Name of the file to backup
    
    :Returns:
        
        out : str
              New unique name for the file to backup or None if no file existed to back up
    '''
    
    arcname = None
    if os.path.exists(filename): 
        base, ext = os.path.splitext(filename)
        if ext == '.zip': ext='.bak.zip'
        else: ext = '.zip'
        zf = zipfile.ZipFile(base+ext, mode='a')
        arcname = os.path.basename(backup_name(filename))
        try:
            zf.write(filename, arcname=arcname, compress_type=zipfile.ZIP_STORED)
        finally: zf.close()
    return arcname

def backup_name(filename):
    ''' Generate a unique name based on the current date for the backup
    version of the specified file.
    
    :Parameters:
    
    filename : str
               Name of the file to backup
    
    :Returns:
        
        out : str
              New unique name for the file to backup
    '''
    
    base, ext = os.path.splitext(filename)
    timpstamp=time.strftime("_%Y_%m_%d_%H_%M_%S",time.localtime(os.path.getctime(filename)))
    return base+timpstamp+ext

def check_options(options):
    '''Check if the option values are valid
    
    This function tests if the option values are valid and performs any
    pre-processing of the options before the program is run.
    
    :Parameters:

    options : object
              Object whose fields are options that hold the corresponding values
    '''
    #from autopart.packrat.options import OptionValueError
    
    pass

if not loaded:
    loaded = True
    
    class NullHandler(logging.Handler):
        ''' Logging handler that does nothing
        '''
        def emit(self, record):
            ''' Dummy emit function
            
            :Parameters:
            
            record : object
                     Record to log
            '''
            pass
    h = NullHandler()
    logging.getLogger("vispider").addHandler(h)

class ExceptionFilter(logging.Filter):
    '''Disallows exceptions to be logged
    '''

    def filter(self, record):
        ''' Disallow exceptions to be logged
        
        :Parameters:
        
        record : LogRecord
                 Current log record
        
        :Returns:
            
            val : bool
                  True if record does not contain an exception
        '''
        
        #print "here: ", record.exc_info
        #record.exc_info  = None
        return record.exc_info is None
