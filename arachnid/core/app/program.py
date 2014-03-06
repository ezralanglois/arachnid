''' Defines the program architecture for every application

This module defines a generic architecture shared among many scripts as well as providing
shared utilities such as option parsing, option checking, option updating, MPI setup, 
multi-threading control, input filename updates, configuration file creation, logging setup,
file processing and version control.

.. beg-dev

Essentially a script can utilize one of two program architectures:

    #. Batch processing
    #. Group or file processing

This is determined by whether the script defines a :py:func:`process` or :py:func:`batch`
entry point.

By convention, a program should define a main entry point for execution from outside the
script as follows:

.. sourcecode:: py

    def main():
        #Main entry point for this script
        from arachnid.core.app import program 
        program.run_hybrid_program(__name__, description = """Description of the script""")

    if __name__ == "__main__": main()

Basically, it should define a `main` function that can serve as an entry point for
`setup.py`. And, in case the script is executed from the command line, it should contain
the following:

.. sourcecode:: py

    if __name__ == "__main__": main()

Interface
---------

The target module must define one of the following functions:

.. py:function:: process(filename, **extra)

   Process a single file. During parallel processing this is invoked by a worker process.
   
   See :py:mod:`arachnid.core.app.file_processor` for more information on the file or group processing 
   architecture. 

   :param filename: Input filename to process
   :param extra: Unused keyword arguments (Options from the command line or config file, plus additional options)
   :returns: At minimum filename, however a tuple of additional arguments can be returned for processing by :py:func:`reduce_all`
   
.. py:function:: batch(files, **extra)

   Process all input files.

   :param files: List of input filenames to process
   :param extra: Unused keyword arguments (Options from the command line or config file, plus additional options)
   
The target module must define all the the following:

.. py:function:: setup_options(parser, pgroup=None, main_option=False)

   Setup program options for command-line and configuration file

   :param parser: OptionParser
   :param pgroup: Root OptionGroup
   :param main_option: If true, add the options specific to running this script

The target module may include any of the following functions:

.. py:function:: change_option_defaults(parser)

   Customize default options in other modules for this specific script.

   :param parser: OptionParser

.. py:function:: dependents()

   Return a list of dependent modules and add their options. Dependents of a dependent will also be
   recursively found and added.

   :returns: List of dependent modules

.. py:function:: update_options(options)

   Utilize parsed options and add additional option values.

   :param options: object containing options as fields

.. py:function:: check_options(options, main_option=False)

   Test the validity of each option value and throw OptionValueError if
   an invalid value is found.

   :param options: object containing options as fields
   :param main_option: If true, add the options specific to running this script

.. py:function:: flags()

   Return a dictionary of support properties

   :returns: Dictionary of supported properties

.. end-dev

Parameters
----------

The following options are, by convention, defined in most scripts with
a few exceptions.

.. program:: standard

.. beg-convention-options

.. option:: -i <FILENAME1,FILENAME2>, --input-files <FILENAME1,FILENAME2>, FILENAME1 FILENAME2
    
    List of filenames for the input files. If you use the parameters `-i` or `--inputfiles` they must be
    comma separated (no spaces). If you do not use a flag, then separate by spaces. For a very large number of files
    (>5000) use `-i "filename*"`
    
.. option:: -o <FILENAME>, --output <FILENAME>
    
    Filename for the output file with correct number of digits (e.g. vol_0000.spi)

.. end-convention-options

The following parameters are added to a script using the program architecture. 

.. program:: shared

.. beg-program-options

.. option:: -z <FILENAME>, --create-cfg <FILENAME>
    
    Create a configuration file (if the value is 1, true, y, t, then write to standard output)

.. option:: --prog-version <version>
    
    Version of the program to use, use `latest` to always use the latest version

.. option:: -X, --display-gui <BOOL>
    
    Display the graphical user interface

.. end-program-options
.. beg-openmp-options

.. option:: -t, --thread-count <INT>
    
    Number of threads per machine, 0 means determine from environment (Default: 0)
    
.. end-openmp-options
.. beg-mpi-options

.. option:: --use-MPI <BOOL>
    
    Set this flag True when using mpirun or mpiexec (Only available for MPI-enabled programs)

.. option:: --shared-scratch <FILENAME>
    
    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --home-prefix <FILENAME>
    
    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --local-scratch <FILENAME>
    
    File directory on local node to copy files (optional but recommended for MPI jobs)

.. option:: --local-temp <FILENAME>
    
    File directory on local node for temporary files (optional but recommended for MPI jobs)

.. end-mpi-options

.. todo:: if not exist set to empty home-prefix, local-scratch, local-temp and warn

.. seealso:: 

    Module :py:mod:`arachnid.core.app.file_processor`
        File processing program architecture
    Module :py:mod:`arachnid.core.app.settings`
        Program options parsing for command line and configuration file
    Module :py:mod:`arachnid.core.app.tracing`
        Logging controls

.. Created on Oct 14, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import tracing
import settings
from ..parallel import mpi_utility, openmp
try:
    from ..gui import AutoGUI as autogui
    autogui;
except: autogui=None
import logging, sys, os, traceback,psutil
import arachnid as root_module # TODO: This needs to be found in run_hybrid_program
import file_processor
import multiprocessing

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def run_hybrid_program(name, **extra):
    ''' Main entry point for the program architecture
    
    This function proceeds as follows:
    
        #. Determine module from name
        #. Determine if module supports file/group 
            or batch processing
        #. Collect options, match to values and test validity
           using :py:func:`parse_and_check_options`
        #. Set number of threads for OpenMP (optional)
        #. Invoke entry point to file/group or batch processor
    
    :Parameters:
        
    name : str
           Name of calling module (usually __main__)
    extra : dict
            Pass on keyword arguments
    '''
    
    #_logger.addHandler(logging.StreamHandler())
    main_module = determine_main(name)
    main_template = file_processor if file_processor.supports(main_module) else None
    if hasattr(main_module, 'flags'): extra.update(main_module.flags())
    use_version = extra.get('use_version', False)
    if not use_version and main_template == file_processor: extra['use_version']=True
    #supports_OMP = extra.get('supports_OMP', False)
    _logger.debug("Checking options ...")
    try:
        args, param, parser = parse_and_check_options(main_module, main_template, **extra)
    except SystemExit:
        return
    except VersionChange:
        main_module.main()
        return
    except settings.OptionValueError:
        sys.exit(10)
    except:
        _logger.exception("Unexpected error occurred")
        raise
    _logger.debug("Checking options ... finished.")
    launch_program(main_module, main_template, args, param, parser, **extra)
    
def launch_program(main_module, main_template, args, options, parser, supports_OMP=False, supports_MPI=False, use_version=False, max_filename_len=0, **extra):
    '''
    '''
    
    param = vars(options)
    if hasattr(main_module, 'flags'): 
        extra.update(main_module.flags())
        supports_OMP=extra.get('supports_OMP', False)
        use_version=extra.get('use_version', False)
    mpi_utility.mpi_init(param, **param)
    #_logger.removeHandler(_logger.handlers[0])
    tracing.configure_logging(**param)
    '''
    # do not use these anymore - consider removing
    for org in dependents:
        try:
            if hasattr(org, "organize"):
                args = org.organize(args, **param)
        except:
            _logger.error("org: %s"%str(org.__class__.__name__))
            raise
    '''
    #param = vars(options)
    if param['rank'] == 0 and use_version: 
        val = parser.version_control(options)
        if val is not None: param['vercontrol'], param['opt_changed'] = val
    else: param['opt_changed']=False
    param['file_options'] = parser.collect_file_options()
    param['infile_deps'] = parser.collect_dependent_file_options(type='open')
    param['outfile_deps'] = parser.collect_dependent_file_options(type='save')
    extra['file_options']=param['file_options']
    param.update(update_file_param(**param))
    args = param['input_files'] #options.input_files
    
    if mpi_utility.is_root(**param):
        _logger.info("Program: %s"%(main_module.__name__))# , extra=dict(tofile=True))
        _logger.info("Version: %s"%(str(root_module.__version__)), extra=dict(tofile=True))
        _logger.info("PID: %d"%os.getpid())
        _logger.info("Created: %d"%psutil.Process(os.getpid()).create_time)
    
    #mpi_utility.mpi_init(param, **param)
    if supports_OMP:
        if openmp.get_max_threads() > 0:
            _logger.info("Multi-threading with OpenMP - enabled")
        else:
            if openmp.get_max_threads() >= 0:
                _logger.info("Multi-threading with OpenMP - disabled - %d"%openmp.get_max_threads())
            else:
                _logger.info("Multi-threading with OpenMP - not compiled - %d"%openmp.get_max_threads())
    if supports_OMP and openmp.get_max_threads() > 0:
        if param['thread_count'] > 0:
            openmp.set_thread_count(param['thread_count'])
            _logger.info("Multi-threading with OpenMP - set thread count to %d"%openmp.get_max_threads())
            if param['thread_count'] > multiprocessing.cpu_count():
                _logger.warn("Number of threads exceeds number of cores: %d > %d"%(param['thread_count'], multiprocessing.cpu_count()))
        elif 'worker_count' in param and param['worker_count'] > 1:
            openmp.set_thread_count(1)
    else:
        _logger.warn("Script does not support OpenMP - set OMP_NUM_THREADS in environment (otherwise developer needs to set supports_OMP in the script)", extra=dict(tofile=True))
        
    see_also="\n\nSee .%s.crash_report for more details"%os.path.basename(sys.argv[0])
    try:
        if main_template is not None: 
            _logger.debug("Running template ...")
            main_template.main(args, main_module, **param)
            _logger.debug("Running template ... finished.")
        else: 
            _logger.debug("Running batch ...")
            main_module.batch(args, **param)
            _logger.debug("Running batch ... finished.")
    except IOError, e:
        _logger.error("***"+str(e)+see_also)
        _logger.exception("Ensuring exception logged")
        sys.exit(1)
    except:
        exc_type, exc_value = sys.exc_info()[:2]
        _logger.error("***Unexpected error occurred: "+traceback.format_exception_only(exc_type, exc_value)[0]+see_also)
        _logger.exception("Unexpected error occurred")
        sys.exit(1)
        
def collect_file_dependents(main_module, config_path=None, **extra):
    ''' Collect all filename options into input and output dependents
    
    This function collects all filename options and divides them
    into two groups: input and output.
    
    .. seealso::
    
        Function :py:func:`setup_parser`
            Peforms option collection
    
    :Parameters:
    
    main_module : module
                   Reference to main module
    config_path : str, optional
                  Path to write configuration file
    extra : dict
            New default values for the options
                   
    :Returns:
    
    config_file : str
                  Name of the configuration file
    input_files : list
                 List of options that do not belong to a group
    output_files : list
                    List of option groups
    '''
    
    main_template = file_processor if file_processor.supports(main_module) else None
    if hasattr(main_module, 'flags'): extra.update(main_module.flags())
    parser = setup_parser(main_module, main_template, **extra)[0]
    parser.change_default(**extra)
    
    name = main_module.__name__
    off = name.rfind('.')
    if off != -1: name = name[off+1:]
    if config_path is not None:
        output = os.path.join(config_path, name+".cfg")
    else: output = name+".cfg"
    
    return output, parser.collect_dependent_file_options(type='open', required=True, key='_long_opts'), parser.collect_dependent_file_options(type='save', key='_long_opts')

class program(object):
    '''
    '''
    
    __slots__=('main_module', 'main_template', 'dependents', 'parser', 'config_file', 'values', 'external_prog')
    
    def __init__(self, main_module, main_template, dependents, parser, config_file="", values=None, external_prog=""):
        '''
        '''
        
        self.main_module = main_module
        self.main_template=main_template
        self.dependents = dependents
        self.parser=parser
        self.values = parser.get_default_values() if values is None else values
        self.config_file=config_file
        self.external_prog = external_prog
        
    def program_name(self):
        '''
        '''
        
        return self.external_prog
    
    def configFile(self):
        '''
        '''
        
        return self.config_file
    
    def write_config(self):
        '''
        '''
        
        if self.config_file != "":
            if not os.path.exists(os.path.dirname(self.config_file)):
                os.makedirs(os.path.dirname(self.config_file))
            self.parser.write(self.config_file, values=self.values)
    
    def update(self, param):
        '''
        '''
        
        for key, val in param.iteritems():
            setattr(self.values, key, val)
        
    def name(self):
        '''
        '''
        
        name = self.main_module.__name__
        idx = name.rfind('.')
        if idx != -1: name = name[idx+1:]
        return name
    
    def id(self):
        ''' Get the full name of the main module
        '''
        
        return self.main_module.__name__
    
    def check_options_validity(self):
        '''
        '''
        
        check_options(self.main_module, self.main_template, self.dependents, self.values)
    
    def launch(self):
        '''
        '''

        if len(self.values.input_files) == 1:
            self.values.input_files = self.values.input_files.make(self.values.input_files)
        launch_program(self.main_module, self.main_template, self.values.input_files, self.values, self.parser)
    
    def settings(self):
        '''
        '''
        
        return self.parser.get_config_options(), self.parser.option_groups, self.values
    
    def ensure_log_file(self):
        '''
        '''
        
        if self.values.log_file == "":
            self.values.log_file = os.path.basename(sys.argv[0])+".log"

def generate_settings_tree(main_module, config_path=None, **extra):
    ''' Collect options, groups and values
    
    This function collects all options and option groups then updates their default 
    values, and then returns the options, groups and values.
    
    .. seealso::
    
        Function :py:func:`setup_parser`
            Peforms option collection
    
        Function :py:func:`map_module_to_program`
            Determines script name for module
    
    :Parameters:
    
    main_module : module
                   Reference to main module
    extra : dict
            New default values for the options
                   
    :Returns:
    
    options : list
              List of options that do not belong to a group
    option_groups : list
                    List of option groups
    values : object
             Object where each field is named for an option 
             and holds is corresponding value
    '''
    
    main_template = file_processor if file_processor.supports(main_module) else None
    if hasattr(main_module, 'flags'): extra.update(main_module.flags())
    external_prog = map_module_to_program(main_module.__name__)
    if 'description' in extra:
        extra['description'] = extra['description'].replace('%prog', '%(prog)s')%dict(prog=external_prog)
    parser, dependents = setup_parser(main_module, main_template, external_prog=external_prog, **extra)
    parser.change_default(**extra)
    name = main_module.__name__
    off = name.rfind('.')
    if off != -1: name = name[off+1:]
    if config_path is not None:
        output = os.path.join(config_path, name+".cfg")
    elif hasattr(parser.get_default_values(), 'config_path'):
        output = os.path.join(parser.get_default_values().config_path, name+".cfg")
    else: output = name+".cfg"
    options = parser.parse_file(fin=output) if os.path.exists(output) else None
    return program(main_module, main_template, dependents, parser, output, options, external_prog)
        
def write_config(main_module, config_path=None, **extra):
    ''' Write a configuration file
    
    This function collects all options and option groups then updates their default 
    values, and then write a configuration file
    
    .. seealso::
    
        Function :py:func:`setup_parser`
            Peforms option collection
    
        Function :py:func:`map_module_to_program`
            Determines script name for module
    
    :Parameters:
    
    main_module : module
                   Reference to main module
    config_path : str, optional
                  Path to write configuration file
    extra : dict
            New default values for the options
                   
    :Returns:
    
    output : str
             Output filename of the configuration file
    '''
    
    
    
    #_logger.addHandler(logging.StreamHandler())
    main_template = file_processor if file_processor.supports(main_module) else None
    if hasattr(main_module, 'flags'): extra.update(main_module.flags())
    external_prog=None
    if 'description' in extra:
        external_prog = map_module_to_program(main_module.__name__)
        description = extra['description']
        description=["   "+s.strip()[1:] if len(s.strip())>0 and s.strip()[0] == '-' else '# '+s.strip() for s in description.split("\n")]
        description="\n".join([s for s in description])
        description = description.replace('%prog', '%(prog)s')
        extra['description'] = description%dict(prog=external_prog)
    parser = setup_parser(main_module, main_template, external_prog=external_prog, **extra)[0]
    parser.change_default(**extra)
    name = main_module.__name__
    off = name.rfind('.')
    if off != -1: name = name[off+1:]
    if config_path is not None:
        output = os.path.join(config_path, name+".cfg")
    else: output = name+".cfg"
    parser.write(output) #, options)
    return output

def read_config(main_module, config_path=None, **extra):
    ''' Read in option values from a configuration file
    
    This function collects all options and option groups, and then reads in their values from a configuration file
    
    .. seealso::
    
        Function :py:func:`setup_parser`
            Peforms option collection
    
        Function :py:func:`map_module_to_program`
            Determines script name for module
    
    :Parameters:
    
    main_module : module
                   Reference to main module
    config_path : str, optional
                  Path to write configuration file
    extra : dict
            Unused keyword arguments
                   
    :Returns:
    
    out : dict
          Option/value pairs
    '''
    
    main_template = file_processor if file_processor.supports(main_module) else None
    if hasattr(main_module, 'flags'): extra.update(main_module.flags())
    external_prog=None
    if 'description' in extra:
        external_prog = map_module_to_program(main_module.__name__)
        description = extra['description'].replace('%prog', '%(prog)s')
        extra['description'] = description%dict(prog=external_prog)
    parser = setup_parser(main_module, main_template, external_prog=external_prog, **extra)[0]
    parser.change_default(**extra)
    name = main_module.__name__
    off = name.rfind('.')
    if off != -1: name = name[off+1:]
    if config_path is not None:
        output = os.path.join(config_path, name+".cfg")
    else: output = name+".cfg"
    param = {}
    if os.path.exists(output): param = vars(parser.parse_file(fin=output))
    return param

def update_config(main_module, config_path=None, **extra):
    ''' Test whether a configuration file needs to be updated with new values 
    
    This function collects all options and option groups, then reads in their values from a configuration 
    file and then, tests whether a configuration file needs to be updated with new values.
    
    .. seealso::
    
        Function :py:func:`read_config`
            Peforms option collection and read values from a config file
    
    :Parameters:
    
    main_module : module
                   Reference to main module
    config_path : str, optional
                  Path to write configuration file
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    output : str
             Output filename of the configuration file if it should
             be updated otherwise empty string
    '''
    
    name = main_module.__name__
    '''
    off = name.rfind('.')
    if off != -1: name = name[off+1:]
    if config_path is not None:
        output = os.path.join(config_path, name+".cfg")
    else: output = name+".cfg"
    '''
    param = read_config(main_module, **extra)
    if len(param)==0: return None
    for key,val in param.iteritems():
        '''
        if key not in extra:
            _logger.info("Updating %s config - option %s new"%(name, key))
            return ""
        '''
        if key in extra and str(val) != str(extra[key]):
            _logger.info("Updating %s config - option %s changed from '%s' to '%s'"%(name, key, str(val), str(extra[key])))
            return param
    return None
    
def map_module_to_program(key=None):
    ''' Create a dictionary that maps each module name to 
    its program name
    
    :Parameters:
    
    key : str, optional
          Name of the module to select
    
    :Returns:
    
    map : dict or value
          If key is None then it returns a dictionary 
          mapping each module name to its program, otherwise
          the value for the corresponding key is returned.
    '''
    
    import arachnid.setup
    vals = list(arachnid.setup.console_scripts)
    for i in xrange(len(vals)):
        program, module = vals[i].split('=', 1)
        try:
            module = module.split(':', 1)[0]
        except:
            _logger.error("Module name: %s"%module)
            raise
        vals[i] = (module.strip(), program.strip())
    vals = dict(vals)
    return vals if key is None else vals[key]
        
def setup_parser(main_module, main_template, description="", usage=None, supports_MPI=False, supports_OMP=False, external_prog=None, doc_url=None, **extra):
    ''' Collect all the options from the main module, its dependents, the main template and those shared by all programs
    
    This function also collects all the dependent modules and updates the default values of the options.
    
    .. seealso::
    
        Function :py:func:`setup_program_options`
            Adds additional options to the OptionParser
    
    :Parameters:
        
    main_module : module
                  Main module
    main_template : module
                    Main module wrapper
    description : str
                  Description of the program
    usage : str
            Current usage of the program
    supports_MPI : bool
                   If True, add MPI capability
    supports_OMP : bool
                   If True, add OpenMP capability
    external_prog : str, optional
                    Name of external program to launch script
    doc_url : str, optional
              URL for the documentation, if not specified, then it uses __doc_url__ from the root_module (e.g., arachnid)
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    parser : OptionParser
             OptionParser
    dependents : list
                 List of dependent modules
    '''
    
    
    description=["   "+s.strip()[1:] if len(s.strip())>0 and s.strip()[0] == '.' else '# '+s.strip() for s in description.split("\n")]
    description="\n".join([s for s in description])
        
    dependents = main_module.dependents() if hasattr(main_module, "dependents") else []
    dependents = collect_dependents(dependents)
    if doc_url is None:
        url = root_module.__doc_url__%main_module.__name__ if hasattr(root_module, '__doc_url__') else None
    else: url = doc_url%main_module.__name__
    site_url = None
    if hasattr(root_module, '__url__'): site_url = root_module.__url__
    parser = settings.OptionParser(usage, version=root_module.__version__, description=description, url=url, site_url=site_url, external_prog=external_prog)
    try:
        mgroup = settings.OptionGroup(parser, "Primary", "Options that must be set to run the program", group_order=-20,  id=__name__) 
        main_module.setup_options(parser, mgroup, True)
    except:
        _logger.error("Module name: %s"%str(main_module))
        raise
    parser.add_option_group(mgroup)
    group = settings.OptionGroup(parser, "Dependent", "Options require by dependents of the program", group_order=0,  id=__name__)
    for module in dependents:
        module.setup_options(parser, group)
    if hasattr(main_module, "change_option_defaults"): main_module.change_option_defaults(parser)
    if len(group.option_list) > 0 or len(group.option_groups) > 0: parser.add_option_group(group)
    if main_template == main_module: main_template = None
    setup_program_options(parser, main_template, supports_MPI, supports_OMP)
    return parser, dependents

def parse_and_check_options(main_module, main_template, description="", usage=None, supports_MPI=False, supports_OMP=False, use_version=False, max_filename_len=0):
    ''' Parse the collected options and check option validity
    
    The function also does the following:
        
        #. configures MPI (if necessary)
        #. configures the logging utility
        #. performs option version control
        #. handles config file creation
        #. displays autogui
        
    
    .. seealso::
    
        Function :py:func:`setup_parser`
            Collects options from the module and its dependents
        Function :py:func:`on_error`
            Logs errors and prints usage
        Function :py:func:`update_file_param`
            Updates all filenames with `home_prefix` and/or `local_temp` 
    
    :Parameters:
        
    main_module : module
                  Main module
    main_template : module
                    Main module wrapper
    description : string
                  Description of the program
    usage : string
            Current usage of the program
    supports_MPI : bool
                   If True, add MPI capability
    supports_OMP : bool
                   If True, add OpenMP capability
    use_version : bool
                  If True, add version control capability
    max_filename_len : int
                       Maximum length allowd for filename
    
    :Returns:
    
    args : list
           List of files to process
    options : object
              Dictionary of parameters and their corresponding values
    '''
    
    name = main_module.__name__
    if len(description.split("\n")) > 0:
        name = description.split("\n")[0].strip()
    parser, dependents = setup_parser(main_module, main_template, description, usage, supports_MPI, supports_OMP)
    
    try: options, args = parser.parse_args_with_config()    
    except settings.OptionValueError, inst:
        on_error(parser, inst, parser.get_default_values())
        raise settings.OptionValueError, "Failed when parsing options"
    
    if autogui is not None:
        if 1 == 0:
            if sys.platform == 'darwin': # Hack for my system?
                options=autogui.display_worker(name, parser, options, **vars(options))
            else:
                options=autogui.display_mp(name, parser, options, **vars(options))
            if options is None: sys.exit(0)
            args=list(settings.uncompress_filenames(options.input_files)) #TODO: add spider regexp
        else:
            tracing.configure_logging(**vars(options))
            #program(main_module, main_template, dependents, parser, output)
            #main_module, main_template, dependents, parser, config_file
            autogui.display(program(main_module, main_template, dependents, parser, values=options), **vars(options))
        # Disable automatic version update
        #if options.prog_version != 'latest' and options.prog_version != root_module.__version__: reload_script(options.prog_version)
        #
        #
    
    #parser.write("."+parser.default_config_filename(), options)
    if hasattr(main_module, "update_options"): main_module.update_options(options)
    if options.create_cfg != "":     
        if len(parser.skipped_flags) > 0:
            _logger.warn("The following options where skipped in the old configuration file, you may need to update the new option names in config file you created")
            for flag in parser.skipped_flags:
                _logger.warn("Skipped: %s"%flag)
        if options.create_cfg == '0' or options.create_cfg.lower() == 'false' or options.create_cfg.lower() == 'n' or options.create_cfg.lower() == 'f':
            pass
        elif options.create_cfg == '1' or options.create_cfg.lower() == 'true' or options.create_cfg.lower() == 'y' or options.create_cfg.lower() == 't':
            parser.write(values=options)
            if not hasattr(options, 'noexit'): sys.exit(0)
        else:
            parser.write(options.create_cfg, options)
            if not hasattr(options, 'noexit'): sys.exit(0)
    
    check_options(main_module, main_template, dependents, options, parser)
    return args, options, parser
        
        
        

def check_options(main_module, main_template, dependents, options, parser=None):
    ''' Check the validity of the options
    
    :Parameters:
        
    main_module : module
                  Main module
    main_template : module
                    Main module wrapper
    dependents : list
                 List of dependent modules
    options : object
              Dictionary of parameters and their corresponding values
    parser : OptionParser
             OptionParser
    
    :raises: OptionValueError
    '''
    
    additional = [tracing]
    if main_template is not None and main_template != main_module: additional.append(main_template)
    try:
        for module in dependents+additional:
            if not hasattr(module, "update_options") or module==main_module: continue
            module.update_options(options)
        if parser is not None: parser.validate(options)
        for module in dependents+additional:
            if not hasattr(module, "check_options"): continue 
            module.check_options(options)
        if hasattr(main_module, "check_options"): main_module.check_options(options, True)
    except settings.OptionValueError, inst:
        if parser is not None: on_error(parser, inst, options)
        raise 
        #settings.OptionValueError, "Failed when testing options"
        #parser.error(inst.msg, options)

def on_error(parser, inst, options):
    ''' Called when an OptionValueError has been thrown
    
    This function logs the current error and then prints
    usage information.
    
    :Parameters:
    
    parser : OptionParser
             OptionParser
    inst : Exception
           Exception object with message
    options : object
              Object where each field is named for an option 
              and holds is corresponding value
    '''
    
    param = vars(options)
    print
    tracing.configure_logging(**param)
    _logger.error(inst.msg)
    print parser.get_usage(), "See %s for more information regarding this error"%tracing.default_logfile(**param)
    print

def update_file_param(max_filename_len=0, warning=False, file_options=0, home_prefix=None, local_temp="", **extra):
    ''' Create a soft link to the home_prefix and change all filenames to
    reflect this short cut.
    
    .. note::
    
        This function is only necessary for pySPIDER - should probably be moved there.
    
    :Parameters:
    
    max_filename_len : int
                       Maximum length allowed for filename (0 disables test)
    warning : bool
              Print warning if cannot update
    file_options : list
                   List of `dest` for the file options
    home_prefix : str
                  File path to input files on master node
    local_temp : str
                  File path to temporary directory on each slave node
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    param : dict
            Updated keyword arguments
    '''
    
    param = {}
    if home_prefix is None or local_temp == "" or not os.path.exists(local_temp): 
        
        if warning:
            # Test if file does not exist and is relative path, update
            # based on config file path, if exists
            if home_prefix is None: _logger.warn("--home-prefix not set - no file update")
            if local_temp == "": _logger.warn("--local-temp not set - no file update")
            if not os.path.exists(local_temp): _logger.warn("--local-temp does not exist - no file update")
        
        if home_prefix is not None and home_prefix != "":
            for opt in file_options:
                if opt not in extra or len(extra[opt])==0: continue
                if hasattr(extra[opt], 'append'):
                    param[opt] = []
                    for filename in extra[opt]:
                        if not os.path.isabs(filename):
                            filename = os.path.join(home_prefix, filename)
                        param[opt].append(filename)
                elif not os.path.isabs(extra[opt]):
                    param[opt] = os.path.join(home_prefix, extra[opt])
        return param
    if home_prefix == "":
        home_prefix = os.path.commonprefix([extra[opt] for opt in file_options])
        if not os.path.exists(home_prefix): home_prefix = os.path.dirname(home_prefix)
    base = os.path.basename(home_prefix)
    shortcut = os.path.join(local_temp, "lnk%d_%s"%(mpi_utility.get_rank(**extra),base))
    
    try:
        if os.path.exists(shortcut): os.remove(shortcut)
    except: pass
    else:
        try: os.symlink(home_prefix, shortcut)
        except:
            _logger.error("Src: %s"%home_prefix)
            _logger.error("Des: %s"%shortcut)
            raise
    for opt in file_options:
        if opt not in extra or len(extra[opt])==0: continue
        assert(opt != 'home_prefix')
        if hasattr(extra[opt], 'append'):
            param[opt] = []
            for filename in extra[opt]:
                if len(home_prefix) < len(filename) and filename.find(home_prefix) >= 0:
                    filename = os.path.join(shortcut, filename[len(home_prefix)+1:])
                elif not os.path.isabs(filename):
                    filename = os.path.join(shortcut, filename)
                #if not os.path.exists(filename): raise IOError, "Cannot find file: %s -- %s -- %s"%(filename, shortcut, local_temp)
                if max_filename_len > 0 and len(filename) > max_filename_len:
                    raise ValueError, "Filename exceeds %d characters for %s: %d -> %s"%(opt, max_filename_len, len(filename), filename)
                param[opt].append(filename)
            if len(param[opt]) == 0: del param[opt]
        else:
            if len(home_prefix) < len(extra[opt]) and extra[opt].find(home_prefix) >= 0:
                param[opt] = os.path.join(shortcut, extra[opt][len(home_prefix)+1:])
            elif not os.path.isabs(extra[opt]):
                param[opt] = os.path.join(shortcut, extra[opt])
            else: continue
            #if not os.path.exists(param[opt]): 
            # _logger.warn("Cannot find file: %s -- %s -- %s"%(param[opt], shortcut, local_temp))
            #raise IOError, "Cannot find file: %s -- %s -- %s"%(param[opt], shortcut, local_temp)
            if max_filename_len > 0 and len(param[opt]) > max_filename_len:
                raise ValueError, "Filename exceeds %d characters for %s: %d -> %s"%(opt, max_filename_len, len(extra[opt]), extra[opt])
    return param

def setup_program_options(parser, main_template, supports_MPI=False, supports_OMP=False):
    # Collection of options necessary to use functions in this script
    
    parser.add_option("",   create_cfg="",          help="Create a configuration file (if the value is 1, true, y, t, then write to standard output)", gui=dict(nogui=True))
    gen_group = settings.OptionGroup(parser, "General", "Options to general program features",  id=__name__)
    prg_group = settings.OptionGroup(parser, "Program", "Options to program features",  id=__name__)
    prg_group.add_option("",   prog_version=root_module.__version__, help="Select version of the program (set `latest` to use the lastest version`)")
    if supports_MPI and mpi_utility.supports_MPI():
        group = settings.OptionGroup(parser, "MPI", "Options to control MPI",  id=__name__, dependent=False)
        group.add_option("",   use_MPI=False,          help="Set this flag True when using mpirun or mpiexec")
        group.add_option("",   shared_scratch="",      help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="save"), dependent=False)
        group.add_option("",   home_prefix="",         help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="open"), dependent=False)
        group.add_option("",   local_scratch="",       help="File directory on local node to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="save"), dependent=False)
        group.add_option("",   local_temp="",          help="File directory on local node for temporary files (optional but recommended for MPI jobs)", gui=dict(filetype="save"), dependent=False)
        gen_group.add_option_group(group)
    if supports_OMP:# and openmp.get_max_threads() > 1:
        prg_group.add_option("-t",   thread_count=1, help="Number of threads per machine, 0 means determine from environment", gui=dict(minimum=0), dependent=False)
    tracing.setup_options(parser, gen_group)
    if autogui is not None:
        autogui.setup_options(parser, gen_group)
    if main_template is not None: main_template.setup_options(parser, gen_group)
    gen_group.add_option_group(prg_group)
    parser.add_option_group(gen_group)

"""
def reload_script(version):
    ''' Update sys.path and reload all the modules
    
    :Parameters:
        
    version : str
              The version of the modules to use
    '''
    
    if version != root_module.__version__:
        _logger.warn("Version mismatch: %s != %s - loading version %s of the code"%(version, root_module.__version__, version))
        curr_package = os.path.dirname(root_module.__path__[0])
        index = curr_package.find(root_module.__version__)
        if index == -1:
            logging.warn("Cannot update version - installation does not support this")
            return
        
        next_package = curr_package[:index]+version+curr_package[index+len(root_module.__version__):]
        if not os.path.exists(next_package):
            logging.warn("Cannot update version - the version requested %s is not installed"%version)
            return
        
        for index in xrange(len(sys.path)):
            if sys.path[index] == curr_package: break
        try:
            sys.path[index] = next_package
        except:
            logging.warn("Bug: Cannot find current package: %s"%curr_package)
    
        modules = [key for key in sys.modules.keys() if key.find(root_module.__name__)==0]
        saved = None
        for mod in modules:
            if mod.find(__name__) == -1:
                try:
                    mod = __import__(mod)
                except: continue
                if hasattr(mod, '_logger'):
                    for h in mod._logger.handlers: mod._logger.removeHandler(h)
                try:reload(mod)
                except: pass
                if hasattr(mod, '_logger'):
                    print mod, len(mod._logger.handlers)
            else: saved = mod
        assert(saved is not None)
        reload(__import__(saved))
        raise VersionChange
"""

class VersionChange(StandardError):
    ''' Thrown when the application has to make a version change.
    '''
    pass
        
def determine_main(name):
    ''' Find the calling main module
    
    :Parameters:

    name : str
           Name of the calling module
    
    :Returns:
    
    module : Module
             Calling main module
    '''
    
    if name == '__main__':
        import inspect
        main_module = inspect.getmodule(inspect.stack()[1][0])
        #main_module = inspect.getmodule(inspect.stack()[2][0])
        if main_module is None:
            name = os.path.splitext(inspect.stack()[1][1])[0]
            name = name.replace(os.sep, '.')
            main_module = __import__(name, fromlist="*")
        elif main_module.__name__ == '__main__':
            name = os.path.splitext(inspect.stack()[1][1])[0]
            name = name.replace(os.sep, '.')
            while name.find('.') != -1:
                try:
                    main_module = __import__(name, fromlist="*")
                except:
                    name = name.split('.', 1)[1]
                else:
                    if main_module.__name__ != '__main__': break
    else:
        main_module = sys.modules[name]
    return main_module

def collect_dependents(dependents):
    ''' Recursively search for additional dependents and add to current list
    
    :Parameters:
    
    dependents : list
                 List of dependents to search
    
    :Returns:
    
    out : list
          List of dependents with no repeats
    '''
    
    deps = list(dependents)
    index = 0
    while index < len(deps):
        if hasattr(deps[index], "dependents"):
            deps.extend(deps[index].dependents())
        index += 1
    return list(set(deps))


