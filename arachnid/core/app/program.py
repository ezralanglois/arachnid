''' Handles functionality shared by every program in arachnid: option parsing, MPI setup, version control, etc.

A template program defines a generic structure shared among many scripts. In other
words, it defines a common set of options.

Parameters
----------

.. program:: shared

.. option:: --use-MPI <BOOL>
    
    Set this flag True when using mpirun or mpiexec (Only available for MPI-enabled programs)

.. option:: -z <FILENAME>, --create-cfg <FILENAME>
    
    Create a configuration file (if the value is 1, true, y, t, then write to standard output)

.. option:: --prog-version <version>
    
    Version of the program to use, use `latest` to always use the latest version

.. option:: --shared-scratch <FILENAME>
    
    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --home-prefix <FILENAME>
    
    File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)

.. option:: --local-scratch <FILENAME>
    
    File directory on local node to copy files (optional but recommended for MPI jobs)

.. option:: --local-temp <FILENAME>
    
    File directory on local node for temporary files (optional but recommended for MPI jobs)

.. option:: -t, --thread-count <INT>
    
    Number of threads per machine, 0 means determine from environment (Default: 0)

.. Created on Oct 14, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..parallel import mpi_utility, openmp
from ..gui import settings_editor
import tracing, settings
import logging, sys, os, traceback
import arachnid as root_module
import file_processor

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def run_hybrid_program(name, description, usage=None, supports_MPI=True, supports_OMP=False, use_version=False, max_filename_len=0, output_option=None):
    ''' Run the main function or other entry points in the calling module
    
    :Parameters:
        
    name : string
           Name of calling module (usually __main__)
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
    output_option : str, optional
                    If specified, then add `-o, --output` option with given help string
    '''
    
    _logger.addHandler(logging.StreamHandler())
    _logger.debug("Run program: %d, %d"%(supports_MPI, mpi_utility.supports_MPI()))
    main_module = determine_main(name)
    main_template = file_processor if file_processor.supports(main_module) else None
    try:
        parser, args, param = parse_and_check_options(main_module, main_template, description, usage, supports_MPI, supports_OMP, use_version, max_filename_len, output_option)
    except VersionChange:
        main_module.main()
        return
    
    mpi_utility.mpi_init(param, **param)
    if supports_OMP:
        if param['thread_count'] > 0:
            openmp.set_thread_count(param['thread_count'])
    see_also="\n\nSee .%s.crash_report for more details"%os.path.basename(sys.argv[0])
    try:
        if main_template is not None: main_template.main(args, main_module, **param)
        else: main_module.batch(args, **param)
    except IOError, e:
        _logger.error("***"+str(e)+see_also)
        _logger.exception("Ensuring exception logged")
    except:
        exc_type, exc_value = sys.exc_info()[:2]
        _logger.error("***Unexpected error occurred: "+traceback.format_exception_only(exc_type, exc_value)[0]+see_also)
        _logger.exception("Unexpected error occurred")
        sys.exit(1)
        
def write_config(main_module, description="", config_path=None, supports_MPI=False, supports_OMP=False, **extra):
    ''' Write a configuration file to an output file
    
    :Parameters:
    
    main_module : module
                   Reference to main module
    description : str
                  Header on configuration
    config_path : str, optional
                  Path to write configuration file
    supports_MPI : bool
                   Set True if the script supports MPI
    supports_OMP : bool
                   If True, add OpenMP capability
                   
    :Returns:
    
    output : str
             Output filename of the configuration file
    '''
    
    _logger.addHandler(logging.StreamHandler())
    main_template = file_processor if file_processor.supports(main_module) else None
    dependents = main_module.dependents() if hasattr(main_module, "dependents") else []
    dependents = collect_dependents(dependents)
    if main_template is not None and main_template != main_module: dependents.append(main_template)
    dependents.extend([settings_editor])
    parser = setup_parser(main_module, dependents, description%dict(prog=map_module_to_program(main_module.__name__)), None, supports_MPI, supports_OMP, False, None)
    parser.change_default(**extra)
    name = main_module.__name__
    off = name.rfind('.')
    if off != -1: name = name[off+1:]
    if config_path is not None:
        output = os.path.join(config_path, name+".cfg")
    else: output = name+".cfg"
    parser.write(output) #, options)
    return output
    
def map_module_to_program(key=None):
    ''' Create a dictionary that maps each module name to 
    its program name
    
    :Parameters:
    
    key : str, optional
          Name of the module to select
    
    :Returns:
    
    map : dict
          Dictionary mapping each module name to its program
    '''
    
    import arachnid.setup
    vals = list(arachnid.setup.console_scripts)
    for i in xrange(len(vals)):
        program, module = vals[i].split('=', 1)
        module, main = module.split(':', 1)
        vals[i] = (module.strip(), program.strip())
    vals = dict(vals)
    return vals if key is None else vals[key]
        
def setup_parser(main_module, dependents, description="", usage=None, supports_MPI=False, supports_OMP=False, use_version=False, output_option=None):
    '''Parse and setup the parameters for the generic program
    
    :Parameters:
        
    main_module : module
                  Main module
    dependents : list
                 List of dependent modules
    description : string
                  Description of the program
    usage : string
            Current usage of the program
    supports_MPI : bool
                   If True, add MPI capability
    supports_OMP : bool
                   If True, add OpenMP capability
    output_option : str, optional
                    If specified, then add `-o, --output` option with given help string
    
    :Returns:
    
    args : list
           List of files to process
    options : object
              Options object
    '''
    
    parser = settings.OptionParser(usage, version=root_module.__version__, description=description)
    try:
        group = settings.OptionGroup(parser, "Required Parameters", "Options that must be set to run the program", group_order=-20,  id=__name__) 
        main_module.setup_options(parser, group, True)
    except:
        _logger.error("Module name: %s"%str(main_module))
        raise
    if hasattr(main_module, "setup_main_options"): main_module.setup_main_options(parser, group)
    if len(group.option_list) > 0: parser.add_option_group(group)
    for module in dependents:
        module.setup_options(parser, group)
    setup_program_options(parser, supports_MPI, supports_OMP, output_option)
    return parser

def parse_and_check_options(main_module, main_template, description, usage, supports_MPI=False, supports_OMP=False, use_version=False, max_filename_len=0, output_option=None):
    '''Parse and setup the parameters for the generic program
    
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
    output_option : str, optional
                    If specified, then add `-o, --output` option with given help string
    
    :Returns:
        
    parser : OptionParser
             The option parser used to parse the command line parameters
    args : list
           List of files to process
    param : dict
            Dictionary of parameters and their corresponding values
    '''
    
    dependents = main_module.dependents() if hasattr(main_module, "dependents") else []
    dependents = collect_dependents(dependents)
    if main_template is not None and main_template != main_module: dependents.append(main_template)
    dependents.extend([settings_editor])
    
    description="\n#  ".join([s.strip() for s in description.split("\n")])
    parser = setup_parser(main_module, dependents, description, usage, supports_MPI, supports_OMP, use_version, output_option)
    options, args = parser.parse_args_with_config()
    
    if 1 == 0:
        unset = parser.collect_unset_options()
        if len(unset) > 0:
            for opt in unset:
                print opt
            sys.exit(1)
        sys.exit(0)
        
    options.create_cfg = settings_editor.display(parser, **vars(options))
    
    if options.prog_version != 'latest' and options.prog_version != root_module.__version__: reload_script(options.prog_version)
    #parser.write("."+parser.default_config_filename(), options)
    if options.create_cfg != "":     
        if len(parser.skipped_flags) > 0:
            _logger.warn("The following options where skipped in the old configuration file, you may need to update the new option names in config file you created")
            for flag in parser.skipped_flags:
                _logger.warn("Skipped: %s"%flag)
        if options.create_cfg == '0' or options.create_cfg.lower() == 'false' or options.create_cfg.lower() == 'n' or options.create_cfg.lower() == 'f':
            pass
        elif options.create_cfg == '1' or options.create_cfg.lower() == 'true' or options.create_cfg.lower() == 'y' or options.create_cfg.lower() == 't':
            parser.write(values=options)
            sys.exit(0)
        else:
            parser.write(options.create_cfg, options)
            sys.exit(0)
    
    try:
        for module in dependents+[tracing]:
            if not hasattr(module, "update_options"): continue
            module.update_options(options)
        parser.validate(options)
        for module in dependents+[tracing]:
            if not hasattr(module, "check_options"): continue
            module.check_options(options)
        if hasattr(main_module, "check_main_options"): main_module.check_main_options(options)
        if hasattr(main_module, "check_options"): main_module.check_options(options, True)
    except settings.OptionValueError, inst:
        parser.error(inst.msg, options)
    
    param = vars(options)
    rank = mpi_utility.get_rank(**param)
    if rank == 0 and use_version: param['vercontrol'], param['opt_changed'] = parser.version_control(options)
    _logger.removeHandler(_logger.handlers[0])
    tracing.configure_logging(rank=rank, **param)
    for org in dependents:
        try:
            if hasattr(org, "organize"):
                args = org.organize(args, **param)
        except:
            _logger.error("org: %s"%str(org.__class__.__name__))
            raise
    param = vars(options)
    param['file_options'] = parser.collect_file_options()
    param['infile_deps'] = parser.collect_file_options(type='open')
    param['outfile_deps'] = parser.collect_file_options(type='save')
    param.update(update_file_param(max_filename_len, **param))
    args = options.input_files
    return parser, args, param

def update_file_param(max_filename_len, file_options, home_prefix=None, local_temp="", shared_scratch="", local_scratch="", **extra):
    ''' Create a soft link to the home_prefix and change all filenames to
    reflect this short cut.
    
    :Parameters:
    
    max_filename_len : int
                       Maximum length allowed for filename
    file_options : list
                   List of `dest` for the file options
    home_prefix : str
                  File path to input files on master node
    local_temp : str
                  File path to temporary directory on each slave node
    shared_scratch : str
                     File path to shared directory accessible to each slave node (unused here)
    local_scratch : str
                    File path to local scratch directory on each slave node (unused here)
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    param : dict
            Updated keyword arguments
    '''
    
    param = {}
    if home_prefix is None or local_temp == "": 
        
        # Test if file does not exist and is relative path, update
        # based on config file path, if exists
        
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
        if opt not in extra: continue
        if extra[opt].find(home_prefix) != 0: continue
        if isinstance(extra[opt], list):
            for filename in extra[opt]:
                filename = os.path.join(shortcut, filename[len(home_prefix):])
                if max_filename_len > 0 and len(filename) > max_filename_len:
                    raise ValueError, "Filename exceeds %d characters for %s: %d -> %s"%(opt, max_filename_len, len(filename), filename)
                param[opt].append(filename)
        else:
            param[opt] = os.path.join(shortcut, extra[opt][len(home_prefix):])
            if max_filename_len > 0 and len(param[opt]) > max_filename_len:
                raise ValueError, "Filename exceeds %d characters for %s: %d -> %s"%(opt, max_filename_len, len(extra[opt]), extra[opt])
    return param

def setup_program_options(parser, supports_MPI=False, supports_OMP=False, output_option=None):
    # Collection of options necessary to use functions in this script
    
    parser.add_option("",   create_cfg="",          help="Create a configuration file (if the value is 1, true, y, t, then write to standard output)", gui=dict(nogui=True))
    gen_group = settings.OptionGroup(parser, "General", "Options to general program features",  id=__name__)
    gen_group.add_option("",   prog_version=root_module.__version__, help="Select version of the program (set `latest` to use the lastest version`)")
    if output_option is not None:
        parser.add_option("-o", output="", help=output_option, gui=dict(filetype="save"), required_file=True)
    if supports_MPI and mpi_utility.supports_MPI():
        group = settings.OptionGroup(parser, "MPI", "Options to control MPI",  id=__name__, dependent=False)
        group.add_option("",   use_MPI=False,          help="Set this flag True when using mpirun or mpiexec")
        group.add_option("",   shared_scratch="",      help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="save"))
        group.add_option("",   home_prefix="",         help="File directory accessible to all nodes to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="open"))
        group.add_option("",   local_scratch="",       help="File directory on local node to copy files (optional but recommended for MPI jobs)", gui=dict(filetype="save"))
        group.add_option("",   local_temp="",          help="File directory on local node for temporary files (optional but recommended for MPI jobs)", gui=dict(filetype="save"))
        gen_group.add_option_group(group)
    _logger.error("here1")
    if supports_OMP and openmp.get_max_threads() > 1:
        gen_group.add_option("-t",   thread_count=0, help="Number of threads per machine, 0 means determine from environment")
    _logger.error("here2")
    tracing.setup_options(parser, gen_group)
    parser.add_option_group(gen_group)
    
    
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
    '''
    
    deps = list(dependents)
    index = 0
    while index < len(deps):
        if hasattr(deps[index], "dependents"):
            deps.extend(deps[index].dependents())
        index += 1
    return list(set(deps))


