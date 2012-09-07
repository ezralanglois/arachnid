''' Parse options on the command line or in a configuration file

Every program shares a standard option interface that provides a rich set
of features to the user.

Parameters
----------

.. option:: -i <list>, --input-files <list>
    
    Input files in addition to those placed on the command line

.. option:: -c <filename>, --config-file <filename>
    
    Read a configuration file for options

.. option:: --version
    
    Show program's version number and exit

.. option:: -h, --help
    
    Show program's version number and exit

Examples
---------

Command-line
++++++++++++

A basic method to set options for a script is to use the command line. The
standard format for command-line options can be seen in the example below.f
In general, the command is followed by a list of input files, then specific
options can be set.

.. sourcecode:: sh
    
    $ ara-autopick <input-files> --template <name> -o <file-path>
    
    $ ara-autopick mic_001.spi --template reference.spi -o sndc_000.spi
    
There are two components to setting an option(`--metrics precision`):

    - Flag: `--metrics` or `-m`
    - Value: precision
    
Some arguments such as `--metrics` supports a short form (`-m`). Also,
most flags follow the pattern `--name value`, however input files is the
one exception that does not require a flag.

The input files can be explicitly set as follows:

.. sourcecode:: sh
    
    $ ara-autopick <input-files>
    $ ara-autopick --input-files <input-files>
    $ ara-autopick -i <input-files>

Available Options
+++++++++++++++++

The options available for a specific script can be found in a number of ways. First
one can invoke the script with `-h` or `--help`. This option display is best for
learning about command-line options.

.. sourcecode:: sh
    
    $ ara-autopick -h | more
    
    Usage: ara-autopick inputfile1 inputfile2 --option-name value ...
    
    This program estimates the performance of a machine learning or #  particle
    selection algorithm. #
    
    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -i INPUT_FILES, --input-files=INPUT_FILES
                            Input files in addition to those placed on the command
                            line
      -o OUTPUT, --output=OUTPUT
                            Path and name of the output file
    --More--
    
Second, one can invoke the script with no arguments (or an incorrect argument). This 
option is best for learning about configuration file options.

.. sourcecode:: sh
    
    $ ara-autopick | more
    settings:709:root:ERROR - No metrics specified
    #  Program:    ara-autopick
    #  Version:    1.1.1
    
    #  This program estimates the performance of a machine learning or
    #  particle selection algorithm.
    #  
    
    input-files:        #    Input files in addition to those placed on the command line
    output:            #    Path and name of the output file
    --More--
    
Configuration Files
+++++++++++++++++++

A configuration file can be created by running the script without any parameters, as shown above. To
save the configuration file, you can redirect the output stream to a file as follows:

.. sourcecode:: sh
    
    $ mkdir cfg
    $ ara-autopick > cfg/auto.cfg
    $ more cfg/auto.cfg
    #  Program:    ara-autopick
    #  Version:    1.1.1
    
    #  This program estimates the performance of a machine learning or
    #  particle selection algorithm.
    #  
    
    input-files:        #    Input files in addition to those placed on the command line
    output:            #    Path and name of the output file
    --More--
    
Note that the directory creation above (`mkdir cfg`) is not necessary but as will be come apparent, it is
recommended.

A configuration file can be used as follows: 

.. sourcecode:: sh

    $ ara-autopick -c cfg/auto.cfg # Explicitly set filename
    $ ara-autopick -c auto         # Automatically find auto.cfg in cfg

Default values in configuration files can be set using command line options:

.. sourcecode:: sh
    
    $ ara-autopick -o output.txt | more
    settings:709:root:ERROR - No metrics specified
    #  Program:    ara-autopick
    #  Version:    1.1.1
    
    #  This program estimates the performance of a machine learning or
    #  particle selection algorithm.
    #  
    
    input-files:                #    Input files in addition to those placed on the command line
    output:      output.txt     #    Path and name of the output file
    --More--
    
Configuration files can also be copied (almost completely):

.. sourcecode:: sh
    
    $ ara-autopick -c old -i "" > cfg/new.cfg  # A parameter must be set to cause an error, e.g. -i ""

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import optparse, types, logging, sys, os, glob
from operator import itemgetter as _itemgetter
from operator import attrgetter as _attrgetter
import functools 
import datetime
#from compiler.ast import Pass

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARN)

class Option(optparse.Option):
    '''Smarter signature to the option class.
        
    This class defines an Option that provides a cleaner signature, see the following examples:
    
    .. sourcecode:: python
        
        >>> from core.app.settings import *
        >>> parser = OptionParser("program -c config", version="1.0", description="Example Options"))
        # Auto flag format
        parser.add_option("-a", append=False,       help="Append all input files into single stack")
        # Explicit flag format
        parser.add_option("-a", dict(append=False), help="Append all input files into single stack")
        # Choice where write_info has an integer value based on position in tuple
        >>> parser.add_option("",   write_info=('All', 'Average', 'Each'), help="Metric summary verbosity", default=0)
    
    This class also provides the following extensions to various values:
    
        1) Parse an option with a dictionary as a value with the format: key1:value1,key2:value2
        2) Parse an option with a list as a value with the format: value1,value2
        3) Choices of any type
        4) String choices with index values returned (use default keyword with an integer index, e.g. default=0)
        5) Boolean types setup by value
        
    It accomplishes these features with a type registry, which contains a set of functions or values for each
    relevant option keyword: callback (automatically sets action), default, type and action.
    
    It also supports the older optparse format as well as all of the optparse keywords:
    
    .. sourcecode:: python
        
        >>> from core.app.settings import *
        >>> parser = OptionParser("program -c config", version="1.0", description="Example Options"))
        # Older optparse format
        >>> parser.add_option("-c", "--id-column", dest="id_col", default=0, help="Column of the spider id in the seleciton file", type="int")

    The following values cannot be used as auto flag names:
        
        - optparse.Option
        
            - action
            - type
            - dest
            - default
            - nargs
            - const
            - choices
            - callback
            - callback_args
            - callback_kwargs
            - help
            - metavar
        
        - settings.Option
        
            - gui: Parameters for rendering the option in the settings manager
            - archive: Flag to indicate whether filename shall be archived
    
    :Parameters:
        
    args : list
           Positional arguments
    kwargs : dict
             Keyword arguments
    '''
    
    EXTRA_ATTRS = ('gui', 'archive', 'required', 'required_file', 'dependent') #, 'glob_more',
    
    def __init__(self, *args, **kwargs):
        #Construct a smarter option
        
        self._required=False
        self._required_file=False
        self._dependent = True
        choices = None
        flag, value = self.determine_flag(args, kwargs)
        if flag is not None:
            args, choices = self.setup_auto_flag(flag, value, args, kwargs)
        if "gui" in kwargs:
            self.gui_hint = kwargs["gui"]
            if 'operations' in self.gui_hint:
                self.gui_hint['type'] = 'operations'
            elif 'decimals' in self.gui_hint:
                self.gui_hint['type'] = 'float'
            elif 'maximum' in self.gui_hint or 'minimum' in self.gui_hint:
                self.gui_hint['type'] = 'int'
            elif 'filetype' in self.gui_hint:
                self.gui_hint['type'] = self.gui_hint['filetype']
                self.gui_hint['file'] = self.gui_hint['filetype']
            elif 'nogui' in self.gui_hint:
                self.gui_hint['type'] = 'nogui'
            #else:
            #    raise ValueError, "Unsupported gui hint: %s"%str(self.gui_hint)
            del kwargs["gui"]
        elif choices is not None:
            self.gui_hint = dict(type="choices", choices=choices)
        else:
            self.gui_hint = {'unset': True, 'type': kwargs.get('type', 'unset')}
        if "archive" in kwargs:
            self.archive = kwargs["archive"]
            del kwargs["archive"]
        else:
            self.archive = False
        if 'dependent' in kwargs:
            self._dependent = kwargs["dependent"]
            del kwargs["dependent"]
        
        optparse.Option.__init__(self, *args, **kwargs)
        self.choices = kwargs['choices'] if choices is None and 'choices' in kwargs else choices
    
    def is_not_config(self):
        ''' Test if option should be written to the configuration file
        
        :Returns:
        
        val : bool
              True if should not be written to config file
        '''
        
        return self.gui_hint is not None and 'nogui' in self.gui_hint
    
    def validate(self, values, validators):
        ''' Validate this option
        
        :Parameters:
            
        values : object
                 Option value collection
        validators : dict
                     Dictionary of validator objects
        '''
        
        if self.gui_hint is not None and 'type' in self.gui_hint and self.gui_hint['type'] in validators:
            val = validators[self.gui_hint['type']](getattr(values, self.dest, self.default), self.dest)
            if val is not None: setattr(values, self.dest, val)
        
        if self._required:
            val = getattr(values, self.dest, self.default)
            if not val:
                raise OptionValueError, "Option %s requires a value - found empty" % (self._long_opts[0])
            if self._required_file:
                if not isinstance(val, list): val = [val]
                for v in val:
                    if os.path.basename(v) == "":
                        raise OptionValueError, "Option %s requires a filename - found directory" % (self._long_opts[0])
                
        
        if not hasattr(self, "_validate") or self._validate is None: return
        message = self._validate(getattr(values, self.dest, self.default))
        if message:
            raise OptionValueError, "Invalid option %s: %s" % (self._long_opts[0], message)
    
    def setup_auto_flag(self, flag, value, args, kwargs):
        '''Setup the auto flag
        
        This method adds the proper ATTRS to the optparse.Option keyword arguments
        
        :Parameters:
            
        flag : string
              The name of a flag
        value : object
                The value of a flag
        args : list
               A list of arguments
        kwargs : dict
                 A dictionary of keyword arguments
        
        :Returns:
        
        return_val : list
                     A list of choices to set for the option
        '''
        
        # Get Validation
        if "validate" in kwargs:
            self._validate = kwargs["validate"]
            del kwargs["validate"]
        else:
            self._validate = None
        
        if 'required' in kwargs:
            self._required=kwargs["required"]
            del kwargs["required"]
        if 'required_file' in kwargs:
            self._required=kwargs["required_file"]
            self._required_file=kwargs["required_file"]
            del kwargs["required_file"]
        
        # Setup proper flag and add to argument list
        flagname = "--"+flag.replace('_', '-')
        args = tuple(args)
        if flagname not in args:
            args = list(args)
            args.append(flagname)
        args = [arg for arg in args if arg != ""]
        
        # Setup the value destination
        if "dest" not in kwargs: kwargs["dest"] = flag
        
        # Setup keywords based on the registry
        registry, choices = self.get_registry(value, kwargs)
        for key, func in registry.iteritems():
            if key in kwargs: continue
            try: v = func(value) if type(func) == types.FunctionType else func
            except: v = func
            if v is not None or key=='type': kwargs[key] = v
        
        # Setup the action with the default action store
        if "action" not in kwargs: kwargs["action"] = "store" if "callback" not in kwargs else "callback"
        # Setup the action with the default value, the given value
        if "default" not in kwargs: kwargs["default"] = registry['default'](value)
        # Setup the type with the default type, class of the given value
        if "type" not in kwargs:    kwargs["type"] = value.__class__.__name__
        # Setup the default value for an index-based choice
        try:
            int(kwargs["default"])
        except: pass
        else:
            if "callback" in kwargs and kwargs["callback"] == Option.choice_index: kwargs["default"] = kwargs["default"]
        return args, choices
    
    def is_choice_index(self):
        ''' Test if Option is a choice where the value is an index
        
        :Returns:
        
        val : bool
              True if the callback holds Option.choice_index
        '''
        
        if self.action == "callback":
            return self.callback == Option.choice_index
        return False
    
    def get_registry(self, value, kwargs):
        '''Get the registry for the given value type
        
        This function gets the registry based on the value type, unless its a tuple
        and the default value is an integer and the tuple contain strings.
        
        Override this function, if you wish to add more than types to the type registry.
        
        :Parameters:
        
        value : object
                The value whose type will be tested
        kwargs : dict
                 A dictionary of keyword arguments
        
        :Returns:
        
        return_val : tuple
                     The first value holds a dictionary, empty if the type is not found and the second a tuple of choices or None
        '''
        
        TYPE_REGISTRY = {
            types.ListType      : {"callback": Option.str_to_list, "default": lambda v: optlist(v), "type": "string"},
            types.DictType      : {"callback": Option.str_to_dict, "default": lambda v: optdict(v), "type": "string"},
            types.BooleanType   : {"action":   lambda v: "store_false" if v else "store_true", "type": None, "default": lambda v: v},
            types.TupleType     : {"callback": Option.choice_type, "default": lambda v: v[0], "type": lambda v: v[0].__class__.__name__},
            "index_true"        : {"callback": Option.choice_index, "default": 0, "type": lambda v: v[0].__class__.__name__},
        }
        
        choices = None
        key = type(value)
        if key == types.TupleType:
            if isinstance(value[0], str) and "default" in kwargs and isinstance(kwargs["default"], int):
                key = "index_true"
            choices = value
        return TYPE_REGISTRY[key] if key in TYPE_REGISTRY else {"default": lambda v: v}, choices
        
    def determine_flag(self, args, kwargs):
        '''Determine the flag from the keyword arguments
        
        This function determines the flag by looking for a keyword argument
        that does not match ATTRS in either Option class.
        
        :Parameters:
            
        args : list
               A list of arguments
        kwargs : dict
                 A dictionary of keyword arguments
        
        :Returns:
        
        return_val : tuple
                     A tuple holding: return flag, value tuple
        '''
        
        # Search for flag in argument list
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                for flag, value in arg.iteritems():
                    del args[i]
                    return flag, value
        
        # Search for flag in argument dictionary
        for flag in kwargs.iterkeys():
            if flag not in optparse.Option.ATTRS and flag not in Option.EXTRA_ATTRS:
                value = kwargs[flag]
                del kwargs[flag]
                return flag, value
        
        return None, None
    
    @staticmethod
    def str_to_dict(option, flag, value, parser):
        '''Convert dictionary to string
        
        :Parameters:
            
        option : Option
                 The option instance calling the callback
        flag : string
               The options flag as seen on the command line
        value : object
                The command line value of the option, type corresponding to the option type
        parser : OptionParser
                 The current option parser instance
        '''
    
        vals = value.strip().split(',')
        d = optdict([v.strip().split(':') for v in vals])
        setattr(parser.values, option.dest, d)
    
    @staticmethod
    def str_to_list(option, flag, value, parser):
        '''Convert dictionary to string
        
        :Parameters:
            
        option : Option
                 The option instance calling the callback
        flag : string
               The options flag as seen on the command line
        value : object
                The command line value of the option, type corresponding to the option type
        parser : OptionParser
                 The current option parser instance
        '''
        
        vals = value.strip().split(',')
        vals = optlist([convert(v.strip()) for v in vals if v.strip() != ""])
        if len(vals) > 0 and vals[0].find(":") != -1:
            for i in range(len(vals)):
                vals[i] = tuple([convert(v.strip()) for v in vals[i].split(":")])
        setattr(parser.values, option.dest, vals)
    
    @staticmethod
    def choice_index(option, flag, value, parser):
        '''Convert dictionary to string
        
        :Parameters:
                
        option : Option
                 The option instance calling the callback
        flag : string
               The options flag as seen on the command line
        value : object
                The command line value of the option, type corresponding to the option type
        parser : OptionParser
                 The current option parser instance
        '''
        
        index = None
        try:
            index = int(value)
            if index >= len(option.choices): raise Exception, "Dummy"
        except:
            for i, c in enumerate(option.choices):
                if c.lower() == value.lower():
                    index = i
                    break
        if index is None:
            choices = ", ".join(map(repr, option.choices))
            raise optparse.OptionValueError("option %s: invalid choice: %r (choose from %s)" % (flag, value, choices))
        setattr(parser.values, option.dest, index)
    
    @staticmethod
    def choice_type(option, flag, value, parser):
        '''Convert dictionary to string
        
        :Parameters:
            
        option : Option
                 The option instance calling the callback
        flag : string
               The options flag as seen on the command line
        value : object
                The command line value of the option, type corresponding to the option type
        parser : OptionParser
                 The current option parser instance
        '''
    
        type = option.choices[0].__class__
        try:
            val = option.choices[int(value)]
        except:
            try:
                val = type(value)
            except:
                raise optparse.OptionValueError("option %s: cannot convert %r to %s" % (flag, value, type.__name__))
        if val in option.choices:
            setattr(parser.values, option.dest, val)
        else: 
            choices = ", ".join(map(repr, option.choices))
            raise optparse.OptionValueError("option %s: invalid choice: %r (choose from %s)" % (flag, value, choices))

class OptionGroup(optparse.OptionGroup):
    '''Extend the option group to take a tree structure
    
    This class allows option groups to hold other option groups
    as children.
    
    It allows allows for keyword arguments, which add additional attributes 
    to the option group class.
        
    .. sourcecode:: python
            
        >>> from core.app.settings import *
        >>> parser = OptionParser("program -c config", version="1.0", description="Example Options"))
        >>> group = OptionGroup(parser, "Benchmark Performance", "Options to control performance benchmarking",  id=__name__)
        >>> group.add_option("-m", metrics=[], help="List of metrics or metric groups to estimate")
        >>> parser.add_option_group(group)
        
    :Parameters:

    parser : OptionParser
             An instance of the option parser
    title : string
            A title of the group
    description : string
                 A help description of the group
    dependent : bool
                Set false if this option has no bearing on output file creation
    kwargs : dict
             A dictionary of keyword arguments
    '''
    
    def __init__(self, parser, title, description=None, group_order=0, dependent=True, **kwargs):
        '''Initialize an option group
        '''
        
        optparse.OptionGroup.__init__(self, parser, title, description)
        for key, val in kwargs.iteritems(): setattr(self, key, val)
        self.option_groups = []
        self.group_order=group_order
        self._dependent = dependent
        self._child = False
        
    def is_child(self):
        ''' Test if the option group is a child of another option group
        
        :Returns:
        
        val : bool
              True if this option group is a child of another group
        '''
        
        return self._child
    
    def group_titles(self):
        '''Get a tuple of group titles.
        
        :Returns:
    
        val : tuple
              List of group titles
        '''
        
        return tuple([g.title for g in self.option_groups])
    
    def add_option_group(self, group):
        '''Add a group to the tree structure
        
        :Parameters:
    
        group : OptionGroup
               A child option group
        '''
        
        #self.parser.add_option_group(group)
        group._child = True
        self.option_groups.append(group)

class OptionParser(optparse.OptionParser):
    '''Extended option parser
    
    It adds functionality to the original option parser including:
    
        - Parsing a configuration file
        - Writing a configuration file
        - Input files: an option in the configuration file that is equivalent to args on the command line
            This command also supports glob based wild cards for files
    
    .. sourcecode:: python
            
        >>> from core.app.settings import *
        >>> parser = OptionParser("program -c config", version="1.0", description="Example Options"))
        >>> parser.add_option("-m", metrics=[], help="List of metrics or metric groups to estimate")
        >>> (options, args) = parser.parse_args_with_config()
    
    :Parameters:
        
    usage : string
            A usage string for your program
    option_list : list
                  A list of options to add
    option_class : class
                   An option class
    version : string
              A version of program
    conflict_handler : string
                       Name of conflict handler
    description : string
                  Description of program
    formatter : string
                Name of help file formatter
    add_help_option : boolean
                      Whether to add a help option
    prog : string
            The name of your program
    comment : (new) string
              A comment character for configuration file
    separator : (new) string
               A flag/value separator for configuration file
    add_input_files : (new) string or None
                    Add option for input files (globbing in configuration file)
    '''
    
    def __init__(self, usage=None, option_list=None, option_class=Option, version=None, conflict_handler='error', description=None, formatter=None, add_help_option=True, prog=None, comment='#', separator=':', add_input_files="input_files"):
        '''Construct an OptionParser
        '''
        
        #if usage is None: usage = "Usage: %prog inputfile1 inputfile2 --option-name value ..."
        optparse.OptionParser.__init__(self, usage, option_list, option_class, version, conflict_handler, description, formatter, add_help_option, prog)
        self.comment=comment
        self.separator=separator
        self.add_input_files = add_input_files
        self.validators = {}
        self.skipped_flags = [] 
        
    def add_validator(self, key, validator, **extra):
        '''Add a validator and default keyword arguments required by the validator
        
        :Parameters:
        
        key : object
              GUI hint object used to find argument type
        validator : function
                    Python function where first argument takes the value type
        extra : dict
                Keyword arguments passed to the validator
        '''
        
        self.validators[key] = functools.partial(validator, **extra)
        
    def add_option_group(self, group):
        '''Add an option group to the option parser
        
        :Parameters:
        
        group : OptionGroup
                An OptionGroup to add to the OptionParser
        '''
        
        if not group._dependent:
            for opt in group.option_list:
                opt._dependent=False
        
        optparse.OptionParser.add_option_group(self, group)
        for subgroup in group.option_groups:
            optparse.OptionParser.add_option_group(self, subgroup)
    
    def version_control(self, options, output=None):
        ''' Create a version control configuration file and tracks every change.
        
        :Parameters:
        
        options : object
                  An object containing option value pairs
        output : str
                 Name of the output, if not specified, then derived from options.output, if that does not exist then it returns with no error
        
        :Returns:
        
        control : VersionControl
                  Access to the version control configuration file
        '''
        
        now = datetime.datetime.now()
        if output is None:
            if not hasattr(options, 'output'): return
            output = options.output
        
        output = os.path.splitext(output)[0]+".cfg"
        output = os.path.join(os.path.dirname(output), "cfg_"+os.path.basename(output))
        if os.path.exists(output):
            values = vars(self.parse_file(fin=output))
            new_values = vars(options)
            changed=self.get_default_values()
            for key, val in new_values.iteritems():
                if key == '_parser' or val is None: continue
                if val == values[key]: setattr(changed, key, None)
                else: setattr(changed, key, val)
            mode='a'
        else:
            changed = options
            mode='w'
        fout = file(output, mode)
        fout.write("# %s - %s\n"%(now.strftime("%Y-%m-%d %H:%M:%S"), str(self.version)))
        self.write(fout, changed, comments=False)
        fout.close()
        
        dep_opts = set(self.collect_dependent_options())
        dep = [val for val in vars(changed).iterkeys() if val in dep_opts]
        return VersionControl(output), len(dep) > 0
    
    def parse_all(self, args=sys.argv[1:], values=None, fin=None):
        '''Parse configuration file, then command line
        
        :Parameters:
            
        args : list
            A list of command line arguments (override sys.argv[1:])
        values : object (dict)
            An object containing default values
        fin : string or stream
            An input filename or input stream
        
        :Returns:
        
        return_val : tuple 
                     A tuple of options and arguments without flags
        '''
        
        values = self.parse_file(args, values, fin)
        return self.parse_args(args, values)
    
    def parse_args_with_config(self, args=sys.argv[1:], values=None):
        ''' Parse arguments from the command line
        
        :Parameters:
        
        args : list
            A list of command line arguments (override sys.argv[1:])
        values : object (dict)
            An object containing default values
        
        :Returns:
        
        return_val : tuple
                     A tuple of options and arguments without flags
        '''
        
        self.add_option("-c", config_file="", help="Read a configuration file for options", gui=dict(nogui=True))#, archive=True)
        try:
            options=self.parse_args()[0]
        except optparse.OptionError, inst:
            self.error(inst.msg, options)
        config_file = None
        if options.config_file != "":
            config_file = options.config_file
            if config_file.find('.') == -1:
                config_file = config_file + ".cfg"
            if not os.path.exists(config_file):
                config_file = os.path.join("cfg", config_file)
            else: config_file = options.config_file
            if not os.path.exists(config_file): 
                self.error("Cannot find specified configuration file: "+options.config_file, options)
        return self.parse_all(args, values, config_file)
    
    def parse_args(self, args=sys.argv[1:], values=None):
        ''' Parse arguments from the command line
        
        This function also performs option validation.
        
        :Parameters:
        
        args : list
            A list of command line arguments (override sys.argv[1:])
        values : object (dict)
            An object containing default values
        
        :Returns:
        
        return_val : tuple
                     A tuple of options and arguments without flags
        '''
        
        _logger.debug("Checking output file - has output "+str(hasattr(values, 'output')))
        if hasattr(values, 'output'): _logger.debug("Checking output file - value of output = \""+str(values.output)+"\"")
        if hasattr(values, self.add_input_files):_logger.debug("Checking input files - has input "+str(getattr(values, self.add_input_files)))
        (options, args) = optparse.OptionParser.parse_args(self, args, values)
        if hasattr(options, self.add_input_files):
            input_files = getattr(options, self.add_input_files)
            _logger.debug("Checking input files - has input "+str(input_files))
            for f in input_files:
                try: args.index(f)
                except: 
                    if not os.path.exists(f):
                        files = glob.glob(f)
                        if len(files) == 0 and hasattr(values, "local_root"):
                            f = os.path.join(values.local_root, f)
                            if not os.path.exists(f): files = glob.glob(f)
                        if len(files) > 0: args.extend(files)
                        else: args.append(f)
                    else: 
                        args.append(f)
            options.input_files = optlist(args)
            _logger.debug("Checking input files - has input "+str(options.input_files))
        options._parser = self
        return (options, args)
    
    def parse_file(self, args=sys.argv[1:], values=None, fin=None):
        '''Parse a configuration file
        
        :Parameters:
        
        args : list
            A list of command line arguments (override sys.argv[1:])
        values : object (dict)
            An object containing default values
        fin : string or stream
            An input filename or input stream
        
        :Returns:
        
        return_val : object
                    A values object
        '''
        
        if values is None: self.values = self.get_default_values()
        else: self.values = values
        if fin is None: fin = self.default_config_filename()
        if isinstance(fin, str):
            _logger.debug("Open config file: %s - exists %d" % (fin, os.path.exists(fin)))
            if not os.path.exists(fin): return self.values
            fin = file(fin, 'r')
        self.skipped_flags = []
        for line in fin:
            pos = line.find(self.comment)
            if line[0] == ' ': continue
            if pos > -1: line = line[:pos]
            line = line.strip()
            if line == "": continue
            key, val = line.split(self.separator, 1)
            key = key.strip()
            val = val.strip()
            if key == "" or val == "": continue
            key="--"+key
            option = self.get_option(key)
            if option is not None:
                _logger.debug("config parser: %s = %s" % (key, val))
                if option.action == "store_false":
                    if val.lower() == "false": option.process(key, val, self.values, self)
                elif option.action == "store_true":
                    if val.lower() == "true": option.process(key, val, self.values, self)
                else:
                    option.process(key, val, self.values, self)
                if hasattr(self.values, 'output'): _logger.debug("Checking output file - value of output = \""+str(self.values.output)+"\"")
                if hasattr(self.values, self.add_input_files):_logger.debug("Checking input files - has input \""+str(getattr(self.values, self.add_input_files))+"\" -- "+str(option.dest)+" == "+str(getattr(self.values, option.dest)))
            else:
                self.skipped_flags.append(key)
        try: fin+"+"
        except:pass
        else: setattr(self.values, 'config_file', fin)
        return self.values
    
    def change_default(self, **kwargs):
        ''' Change the default value for an option
        
        :Parameters:
            
        kwargs : dict
                 Keyword arguments where name of the option is pair with the new value
        '''
        
        for key, val in kwargs.iteritems():
            if isinstance(val, list): kwargs[key] = optlist(val)
        self.defaults.update(kwargs)
    
    def validate(self, values):
        ''' Validate each option value
        
        This method checks if an option has a validation method and invokes
        it to test whether the given value is valid.
        
        :Parameters:
        
        values : object
                 Option value container
        '''
        
        self._validate_options(self.option_list, values)
        for group in self.option_groups:
            self._validate_options(group.option_list, values)
    
    def _validate_options(self, option_list, values):
        ''' Validate a list of options
        
        This method checks if an option has a validation method and invokes
        it to test whether the given value is valid.
        
        :Parameters:
        
        option_list : list
                      List of options to test
        values : object
                 Option value container
        '''
        
        for option in option_list:
            option.validate(values, self.validators)
    
    def write(self, fout=sys.stdout, values=None, comments=True):
        '''Write a configuration file
        
        :Parameters:
            
        fout : string or stream
            An output filename or output stream
        values : object (dict)
            An object containing default values
        comments : bool
                   Set false to disable writing comments to the config file
        '''
        
        if fout is None: fout = self.default_config_filename()
        if values is None: values = self.get_default_values()
        try:
            if isinstance(fout, str): fout = file(fout, 'w')
        except:
            _logger.warn("Cannot write backup config file")
            return
        
        flag_len = int(self.maximum_flag_length() / 8.0)+2
        if comments: self._write_header(fout)
        self._write_options(fout, self.option_list, values, flag_len, comments)
        groups = sorted(self.option_groups, key=_attrgetter('group_order'))
        for group in groups:
            if comments: self._write_group_header(fout, group)
            self._write_options(fout, group.option_list, values, flag_len, comments)
    
    def archived(self):
        '''Get a collection of archived filenames
        
        :Returns:
        
        val : list
              List of archived settings
        '''
        
        arfiles = []
        arfiles.extend(self._archived(self.option_list))
        for group in self.option_groups:
            arfiles.extend(self._archived(group.option_list))
        return arfiles
    
    def _archived(self, options):
        ''' Collect a list of archived filenames
        
        :Parameters:
    
        options : list
                  List of options
        
        :Returns:
        
        val : list
              List of archived settings
        '''
        
        arfiles = []
        for option in options:
            if option.archive:
                arfiles.append(getattr(self.values, option.dest))
        return arfiles
    
    def maximum_flag_length(self):
        '''Determine the maximum length of a configuration flag
        
        :Returns:
        
        return_val : int
                     Maximum length of flag
        '''
        
        flag_len = self._maximum_flag_length(self.option_list)
        for group in self.option_groups:
            flag_len = self._maximum_flag_length(group.option_list, flag_len)
        return flag_len
    
    def _maximum_flag_length(self, options, flag_len=0):
        '''Determine the maximum length of a configuration flag
        
        :Parameters:
    
        options : list
                  List of options
        flag_len : int
                   Current length of the flag
        
        :Returns:
        
        return_val : int
                     Maximum length of flag
        '''
        
        for option in options:
            if len(option._long_opts) == 0: continue
            l = len(str(option._long_opts[0]))
            if l > flag_len: flag_len = l
        return flag_len
    
    def _write_header(self, fout):
        ''' Write the configuration file header
        
        This method defines for format for the configuration file header.
        
        :Parameters:
            
        fout : string or stream
               Output filename or output stream
        '''
        
        prog = self.prog if self.prog is not None else os.path.basename(sys.argv[0])
        fout.write(self.comment+"  Program:\t"+os.path.basename(prog)+"\n")
        fout.write(self.comment+"  Version:\t"+str(self.get_version())+"\n\n")
        #fout.write(self.comment+"  "+self.get_usage()+"\n")
        if self.description is not None:
            fout.write(self.comment+"  "+self.get_description()+"\n\n")
        
    def _write_group_header(self, fout, group):
        ''' Write the group header
        
        This method defines for format for the option group header.
        
        :Parameters:
            
        fout : string or stream
            An output filename or output stream
        group : OptionGroup
            An option group for the header
        '''
        
        fout.write("\n\n"+self.comment+"  "+group.get_description()+"\n")
    
    def _write_options(self, fout, options, values, flag_len, use_comment=True):
        ''' Write options to file stream
        
        This method defines for format for individual options.
        
        :Parameters:
            
        fout : string or stream
            An output filename or output stream
        options : list
            A list of options
        values : object (dict)
            An instance of the value object
        '''
        
        for option in options:
            if len(option._long_opts) == 0 or option.is_not_config(): continue
            name = option._long_opts[0]
            code = option._short_opts[0] if len(option._short_opts) > 0 else ""
            if option.dest is None: continue
            value = getattr(values, option.dest, None)
            if value is None: continue
            if option.is_choice_index(): value = option.choices[value]
            help = option.help
            flag_sep = "".join(["\t" for i in xrange(flag_len-len(name[1:])/8)])
            try:
                comment = ""
                if use_comment:
                    if code != "": code = "\t(%s)\t"%code
                    comment = "\t"+self.comment+"\t"+code+help
                    if option.choices is not None:
                        comment += ",".join([str(c) for c in option.choices])
                fout.write(name[2:]+self.separator+flag_sep+str(value)+comment+"\n")
            except:
                _logger.error("Cannot write out option: "+name)
                raise
    
    def collect_options(self, test, options=None):
        ''' Collect all the options that point to file names
        
        :Parameters:
        
        test : function
               Function used to select option
        options : list
                  List of options to test
        
        :Returns:
        
        names : list
                List of file option destinations
        '''
        
        optionlist = []
        if options is None:
            optionlist.extend(self.collect_options(test, self.option_list))
            for group in self.option_groups:
                optionlist.extend(self.collect_options(test, group.option_list))
        else:
            for opt in options:
                if test(opt) and opt.dest is not None:
                    optionlist.append(opt.dest)
        return optionlist
    
    def collect_unset_options(self):
        ''' Collect all options where the GUI hint parameter is `unset`
        
        :Returns:
        
        names : list
                List of options with unset GUI hint
        '''
        
        def is_unset(opt):
            return opt.gui_hint['type'] == 'str'
        return self.collect_options(is_unset)
        
    def collect_dependent_options(self):
        ''' Collect all the options that are dependencies of the output
        
        :Returns:
        
        names : list
                List of file option destinations
        '''
        
        def is_dependent(opt):
            return opt._dependent and opt.dest != self.add_input_files
        return self.collect_options(is_dependent)
        
    def collect_file_options(self, type=None):
        ''' Collect all the options that point to file names
        
        :Parameters:
        
        type : str
               Type of the file to collect: open or save
        
        :Returns:
        
        names : list
                List of file option destinations
        '''
        
        if type is None:
            def is_file(opt):
                return hasattr(opt, 'gui_hint') and 'file' in opt.gui_hint
            is_file_test = is_file
        else:
            def is_file_type(opt):
                return hasattr(opt, 'gui_hint') and 'type' in opt.gui_hint and opt.gui_hint['type'] == type
            is_file_test = is_file_type
        return self.collect_options(is_file_test)
    
    def program_name(self):
        ''' Get the current name of the program
        
        :Returns:
        
        return_val : string
                     Name of the program
        '''
        
        return sys.argv[0] if self.prog is None else self.prog
    
    def default_config_filename(self):
        ''' Create default configuration file name
        
        :Returns:
        
        return_val : string
                     Basename of prog with cfg extension
        '''
        
        return os.path.splitext(os.path.basename(self.program_name()))[0]+".cfg"
    
    def error(self, msg, values=None):
        '''Log an error, write configuration file and exit program
        
        :Parameters:
            
        msg : string
            An error message
        values : object
                 Object containing option name/value pairs
        '''
        
        logging.error(msg)
        self.write(values=values)
        sys.exit(10)
    
    def create_property_tree(self, factory, property_class=object, converter=lambda x:x, option_list=None, title=None, option_groups=[], tree=None, order=0):
        ''' Create a property tree used to create a graphical user
        interface
        
        :Parameters:
        
        factory : function
                  Function that builds the properties
        property_class : class
                         Base class of the dynamically created propery class
        converter : function
                    Converts one value type to another
        option_list : list
                      List of options to build the tree from
        title : str
                Name of the group
        option_groups : list
                        List of option groups
        tree : OptionValueMap
               Current property tree
        order : int
                Order to add a branch
        
        :Returns:
        
        tree : OptionValueMap
               Mapping between options and properties
        '''
        
        if option_list is not None:
            if tree is None: tree = OptionValueMap(converter)
            if title is None: title = "Critical"
            try:
                Branch = propertyobject(title.replace(' ', '_'), property_class, factory, option_list, order)
            except:
                _logger.error("title = %s"%str(title))
                raise
            title = title.replace('_', ' ')
            title = title.capitalize()
            setattr(Branch, 'DisplayName', title)
            setattr(Branch, '_children', [])
            branch = Branch()
            order = 0 #len(option_list)
            for group in option_groups:
                self.create_property_tree(factory, property_class, converter, group.option_list, group.title, group.option_groups, branch._children, order)
                #order += 1
            tree.append(branch)
        else:
            tree = None
            if len(self.option_list)>0: tree = self.create_property_tree(factory, property_class, converter, self.option_list)
            groups = sorted(self.option_groups, key=_attrgetter('group_order'))
            for group in groups: 
                if group.is_child(): continue
                tree = self.create_property_tree(factory, property_class, converter, group.option_list, group.title, group.option_groups, tree)
        return tree

class OptionValueError(optparse.OptParseError):
    """Exception raised for errors in parsing values of options
        
    :Parameters:
    
    msg : string
        An error message
    """

    def __init__(self, msg):
        '''Create an exception with a message as a parameter
        '''
        
        optparse.OptParseError.__init__(self, msg)

class optdict(dict):
    ''' A dictionary that produces the proper string representation for an option
    
    This class creates a comma separated string representation of the dictionary items where
    the key and value are separated by a colon.
    
    .. sourcecode:: python
            
        >>> from core.app.settings import *
        >>> optdict({'key1': 1, 'key2':2})
        key2:2,key1:1
        
    :Parameters:

    val : string or dictionary object
        A string to convert to a dictionary or some other dictionary compatible object
    '''
    
    def __init__(self, val=None):
        ''' Creates an option compatible dictionary
        '''
        
        if isinstance(val, str):
            val = val.strip()
            if val == "": 
                dict.__init__(self)
                return
            vals = val.split(',')
            try:
                dict.__init__(self, [v.strip().split(':') for v in vals])
            except:
                _logger.error("Error with "+val)
                raise
        elif val is not None:
            dict.__init__(self, val)
        
    def __str__(self):
        '''String representation of the dictionary
        
        This method creates a comma separated string representation of the dictionary items where
        the key and value are separated by a colon.
        
        :Returns:
        
        return_val : string
                    A string representation of the dictionary
        '''
        
        return ",".join([str(k)+":"+str(v) for k, v in self.iteritems()])
    
    def __repr__(self):
        '''String representation of the dictionary
        
        This method creates a comma separated string representation of the dictionary items where
        the key and value are separated by a colon.
        
        :Returns:
        
        return_val : string
                    A string representation of the dictionary
        '''
        
        return str(self)

class optlist(list):
    '''A list that produces the proper string representation for an option
    
    This class creates a comma separated string representation of the list items.
    
    .. sourcecode:: python
            
        >>> from core.app.settings import *
        >>> optlist(['val_1', 'val_2', 'val_3'])
        val_1,val_2,val_3
    
    '''
    
    def __str__(self):
        '''String representation of the list
        
        This method creates a comma separated string representation of the list items.
        
        :Returns:
        
        return_val : string
                    A string representation of the list
        '''
        
        keys = []
        for vals in self: 
            if isinstance(vals, tuple):
                keys.append(":".join([str(k) for k in vals]))
            else:
                keys.append(str(vals))
        return ",".join(keys)
    
    def __repr__(self):
        '''String representation of the list
        
        This method creates a comma separated string representation of the list items.
        
        :Returns:
        
        return_val : string
                    A string representation of the list
        '''
        
        return str(self)

class Validation:
    ''' Set of functions to validate option values
    '''
    
    def __init__(self): pass
    
    @staticmethod
    def empty_values(val):
        ''' Test whether the given string is empty
        
        :Parameters:
    
        val : list
              List to test
        
        :Returns:
        
        return_val : str
                     "No values given"
        '''
        
        return "No values given" if not val else ""
    
    @staticmethod
    def empty_string(val, message):
        ''' Test whether the given string is empty
        
        :Parameters:
    
        val : str
              String to test
        message : str
                  Return message if string empty
        
        :Returns:
        
        return_val : str
                     Given message if string is empty
        '''
        
        return message if val == "" else ""
    
    @staticmethod
    def empty_filename(val):
        ''' Test whether the given filename is empty
        
        :Parameters:
    
        val : str
              String to test
        
        :Returns:
        
        return_val : str
                     Empty filename
        '''
        
        return Validation.empty_string(val, "Empty filename")

def propertyobject(typename, parenttype, property, options, order):
    '''Create an object with the specified properties
    
    :Parameters:
        
    typename : str
               Name of the class
    parenttype : object
                 Parent object
    property : function
               Property function
    options : list
              List of options
    order : int
            Order the object was added
    
    :Returns:
    
    result : object
             Property object
    '''
    
    typename = typename.replace('-', '_')
    parenttype_name = parenttype.__name__
    template = '''class %(typename)s(%(parenttype_name)s):
    def __init__(self): 
        %(parenttype_name)s.__init__(self)\n\n''' % locals()
    if parenttype_name: pass
    
    template += "        self.order_index=%d\n"%order
    for option in options:
        if option.is_not_config() or option.dest is None: continue
        #if isinstance(option.default, str) or (isinstance(option.default, optlist) and len(option.default) == 0):
        if isinstance(option.default, str) or (isinstance(option.default, optlist)):
            template += "        self._m_%s='%s'\n"%(option.dest, str(option.default))
        else:
            template += "        self._m_%s=%s\n"%(option.dest, str(option.default))
    for i, option in enumerate(options):
        if option.is_not_config() or option.dest is None: continue
        opttype = option.type
        if opttype == 'string': 
            opttype = 'int' if option.choices is not None and isinstance(option.default, int) else 'QString'
        if opttype is None: opttype = 'bool'
        help = option.help.split('\n', 1)[0]
        template += "    %s = %s(%d, '%s', attrgetter('_m_%s'), lambda self,value: setattr(self, '_m_%s', value), user=True, doc='%s', editorHints=%s)\n"%(option.dest, property.__name__, i, opttype, option.dest, option.dest, help, str(option.gui_hint))
    
    namespace = dict(itemgetter=_itemgetter, attrgetter=_attrgetter)
    namespace[parenttype.__name__] = parenttype
    namespace[property.__name__] = property
    try:
        exec template in namespace
    except SyntaxError, e:
        raise SyntaxError(e.message + ':\n' + template)
    result = namespace[typename]
    return result

class OptionValueMap(list):
    '''List of parameter trees - provides conversion to a dict
    
    :Parameters:
        
    converter : function
                Convert option value
    '''
    
    def __init__(self, converter):
        "Initialize an Option value map"
        
        self.converter = converter
    
    def todict(self, valuemap=None):
        '''Create a dictionary of parameter-value pairs
        
        :Returns:
        
        valuemap : dict
                   Map of parameter/value pairs
        '''
        
        if valuemap is None: valuemap = {}
        self._collect_parameter_value_pairs(self, valuemap)
        return valuemap
    
    def _collect_parameter_value_pairs(self, branches, valuemap):
        "Recursively collect parameter value pairs"
        
        for branch in branches:
            valuemap.update([(key[3:], self.converter(getattr(branch, key))) for key in dir(branch) if key.startswith('_m_')])
            self._collect_parameter_value_pairs(branch._children, valuemap)

class OptionTree(OptionParser):
    '''Collect option values in a tree
    
    :Parameters:
    
    propertyFactory : function or class
                      Function or class that builds a property (if None no properties are added)
    parentClass : type
                  Class type
    '''
    
    def __init__(self, propertyFactory, parentClass=object, converter=lambda x:x):
        "Initialize an option collector"
        
        OptionParser.__init__(self)
        #optparse.OptionParser.__init__(self, Option, 'error', "")
        self.value_roots = OptionValueMap(converter)
        self.propertyFactory = propertyFactory
        self.parentClass = parentClass
        
    def add_option_group(self, g, parent=None):
        '''Add an option group to the collector
        
        :Parameters:
            
        g : OptionGroup
            Group of options
        parent : list
                 Parent list of children
        '''
        if parent is None: parent = self.value_roots
        
        Branch = propertyobject(g.title.replace(' ', '_'), self.parentClass, self.propertyFactory, g.option_list)
        setattr(Branch, 'DisplayName', g.title)
        setattr(Branch, '_children', [])
        branch = Branch()
        for group in g.option_groups:
            self.add_option_group(group, Branch._children)
        parent.append(branch)
    
    def roots(self):
        '''Get the value roots for the trees
        
        :Returns:
    
        val : list
              List of roots
        '''
        
        return self.value_roots

def build_option_tree(setup_options, propertyFactory=None, parentClass=object, converter=lambda x:x):
    '''Build a tree of option values with corresponding properties
    
    :Parameters:
        
    setup_options : function
                    Function that add options to the OptionParser
    propertyFactory : function or class
                      Function or class that builds a property (if None no properties are added)
    parentClass : type
                  Class type
    
    :Returns:
    
    roots : list
            List of property objects
    '''
    
    tree = OptionTree(propertyFactory, parentClass, converter)
    if isinstance(setup_options, list):
        for _setup_options in setup_options:
            _setup_options(tree)
    else:
        setup_options(tree)
    return tree.roots()
    
class VersionControl(object):
    ''' Helper class that writes specific data as a comment
    in the version control configuration file
    
    :Parameters:
    
    filename : str
               Version control configuration file
    '''
    
    def __init__(self, filename):
        # Initialize VersionControl Object
        
        self.filename = filename
    
    def archive(self, message):
        ''' Archive message in version control configuration file
        
        :Parameters:
        
        message : str
                  Message to add as comment to version control configuration file
        '''
        
        now = datetime.datetime.now()
        fout = file(self.filename, 'a')
        fout.write("# %s: %s\n"%(now.strftime("%Y-%m-%d %H:%M:%S"), message))
        fout.close()

def setup_options_from_doc(parser, *args, **kwargs):
    ''' Extract command line options from function documentation
    
    :Parameters:
    
    parser : OptionParser
             The option parser
    args : list
           List of functions or method names
    kwargs : dict
             Optionally contains `classes` a list of classes for
             given method names
    '''
    
    import inspect
    classes = kwargs.get('classes', [])
    if not isinstance(classes, list): classes=[classes]
    for func in args:
        if not callable(func):
            classobj=None
            for cl in classes:
                if hasattr(cl, func):
                    classobj=cl
                    break
            if classobj is None: raise ValueError, "Function not found: %s"%str(func)
            func = getattr(classobj, func)
        args, varargs, varkw, defaults = inspect.getargspec(func)
        doc = func.__doc__.split('\n')
        group_order=0
        for d in doc:
            index=d.find("Order=")
            if index != -1: 
                group_order=int(d[index+6:].strip())
                break
        name = func.__name__ if callable(func) else func
        group = OptionGroup(parser, name, doc[0], group_order=group_order, id=name)
        for i in xrange(1, len(defaults)+1):
            if defaults[-i] is None: continue
            help = parameter_help(args[-i], doc)
            if help is None: raise ValueError, "Cannot find documentation for parameter %s"%args[-i]
            type = parameter_type(args[-i], doc)
            param = {args[-i]: defaults[-i]}
            #print func, args, defaults, doc
            if type.find('noupdate') != -1:
                param['dependent']=False
            if type.find('infile') != -1:
                param.update(gui=dict(filetype="open"))
            if type.find('outfile') != -1:
                param.update(gui=dict(filetype="save"))
            if isinstance(defaults[-i], tuple):
                param.update(default=0)
            group.add_option("", help=help, **param)
        kwargs.get('group', parser).add_option_group(group)

def parameter_type(name, doc):
    ''' Extract the parameter type from the documentation of a function
    
    :Parameters:
    
    name : str
           Name of the parameter
    doc : str
          Function doc string
              
    :Returns:
    
    type : str
           If found, a string type of the parameter, otherwise None
    '''
    
    for i, d in enumerate(doc):
        idx = d.find(name+" : ")
        if idx  != -1:
            return d[idx+len(name)+len(" : "):].strip()
    return None

def parameter_help(name, doc):
    ''' Extract the parameter description from the documentation of a function
    
    :Parameters:
    
    name : str
           Name of the parameter
    doc : str
          Function doc string
              
    :Returns:
    
    help : str
           If found, a string description of the parameter, otherwise None
    '''
    
    for i, d in enumerate(doc):
        if d.find(name+" : ") != -1:
            return doc[i+1].strip()
    return None

def convert(val):
    '''Try to convert string value to numeric value
    
    .. sourcecode:: py
    
        >>> from core.metadata.format_utility import *
        >>> convert("2.0")
        2
        >>> convert("2.1")
        2.1
    
    :Parameters:
    
    val : string
          Value for conversion
    
    :Returns:

    val : numeric
          Numeric (float or int) converted from string
    '''
    
    try: val = int(val)
    except:
        try: 
            val = float(val)
            i = int(val)
            if (val-i) == 0: 
                val = int(val)
        except: pass
    return val

