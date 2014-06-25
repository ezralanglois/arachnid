''' Common logging utilities and setup for all scripts

Every script supports logging various states of progress and this can be controlled
with a common interface.

Usage
-----

.. beg-usage

Logging is both versatile and redundant allowing the user to both configure
the output, yet still retain the most critical information.

The default for the level of logging is `info` (-v3 or --log-level info). This
prints status updates, warnings and errors. Additional information can be
obtained by changing the level to `debug` (-v4 or --log-level debug). See
examples below:

.. sourcecode:: sh
    
    $ ara-autopick -v4                   # Log most including debug information
    $ ara-autopick -v5                   # Log everything including debug information
    $ ara-autopick --log-level 0         # Do not log any messages
    $ ara-autopick --log-level info      # Log only information level or higher
    
    $ ara-autopick --log-file info.log   # Write all log messages to a file 
    $ ara-autopick --log-file info.log --enable-stderr # Write messages to both file and console

The logging framework also supports fine-grained configuration. This can be enabled using
the `--log-config` parameter. See :ref:`config-logging` for more information.

.. note::
    
    An added benefit of using `--log-file` is that old log files (with the same name) will be backed up in a zip
    archive (of the same name but with a '.zip' extension).

Crash Reports
-------------

The logging module creates two loggers be default:

    #. A crash report logger for all exceptions
    #. A standard logger for user-based messages (default error stream)

The crash report filename is constructed from the name of the script
as follows: 

    .<script_name>.crash_report.<0>

where <script_name> is the name of the script
and <0> is its current MPI rank or 0 for non-MPI programs.

For example, consider a non-MPI script called `ara-info`, its
crash report will be called::

    .ara-info.crash_report.0

This file contains any exceptions that are thrown during the execution of the script and is useful for reporting
bugs. Please attach this file when you submit an issue.

Colors
------

.. role:: red

.. raw:: html

    <style> .red {color:red} </style>

The logging module defines an API to color the logging messages. It
first tests whether the terminal supports color. If so, then it
adds color formatting. 

Currently, only the levelname or :red:`ERROR` in error messages to the 
terminal are colored in red.

.. end-usage

.. _config-logging:

Config Logging
--------------

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
    
    $ ara-autopick --log-config log.conf # Configure logging with a file

Parameters
----------

The tracing script has the following inheritable parameters:

.. program:: shared

.. beg-options

.. option:: -v <CHOICE>, --log-level <CHOICE>
    
    Set logging level application wide: 'critical', 'error', 'warning', 'info', 'debug', 'debug_more' or 0-5

.. option:: --log-file <FILENAME>
    
    Set file to log messages

.. option:: --log-config <FILENAME>
    
    File containing the configuration of the application logging

.. option:: --disable-stderr <BOOL>
    
    If true, output will only be written to the given log file

.. end-options

Module
------

.. beg-dev

Standard practice when using this module is to define a logger
for each module as follows:

.. sourcecode:: py

    import logging
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.DEBUG)

Note that you do not need to import this module, only the standard
logging library. 

On the next line, we define a logger for this module
called `_logger`. Passing `__name__` to `getLogger` creates
a logger with a name based on the module.

On the following line, we set the default log level. If you
want debug logging to print in `-v4` then set the level to
`logging.DEBUG`. However, if you want to suppress debug 
logging until the user specifies `-v5` then set the level
to `logging.INFO`.

A message can bypass the standard logger and go directly to
the crash report by setting the `tofile` to `True`. For
example, consider the following:

.. sourcecode:: py

    logging.warn("Some message", extra=dict(tofile=True))

Logging failed imports of optional modules should be done as
follows:

.. sourcecode:: py

    try: 
        from util import _module
        _module;
    except:
        from arachnid.core.app import tracing
        tracing.log_import_error('Failed to load _module.so module - certain functions will not be available', _logger)
        _module = None

This generates a list of failed imports of optional modules, which will be logged at the start of the crash report.

.. end-dev

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import logging.config
import os
import socket
import time
import sys
import zipfile

class Logger(logging.Logger):
    ''' Maintains a list of loggers for this package
    '''
    
    _log_map = set()
    
    def __init__(self, name, level=logging.DEBUG):
        ''' Create a logger
        
        :Parameters:
            
            name : str
                   Name of the logger
            level : int
                    Logging level
        '''
        logging.Logger.__init__(self, name, level)
        Logger._log_map.add(name)

_loaded = False


_log_level_val = ['critical', 'error', 'warning', 'info', 'debug', 'debug_more']
_log_level_map = {'critical':    logging.CRITICAL,
                 'error':       logging.ERROR,
                 'warning':     logging.WARNING,
                 'info':        logging.INFO,
                 'debug':       logging.DEBUG,
                 'debug_more':  logging.DEBUG-1,
                 }
_log_formats = { 'critical': "%(asctime)s %(message)s",
                'error':    "%(asctime)s %(levelname)s %(message)s",
                'warning':  "%(asctime)s %(levelname)s %(message)s",
                'info':     "%(asctime)s %(levelname)s %(message)s",
                'debug':    "%(asctime)s:%(lineno)d:%(name)s:%(levelname)s - %(message)s",
                'debug_more':    "%(asctime)s:%(lineno)d:%(name)s:%(levelname)s - %(message)s" }

_log_import_errors = []

def log_import_error(message, logger=None):
    ''' Add failed import to list
     
    :Parameter:
         
        message : str
                   Message to log
        logger : Logger, optional
                 Specific logger to use
    '''
    
    _log_import_errors.append(message)
    
def setup_options(parser, pgroup=None):
    '''Add options to the given option parser
    
    :Parameters:
        
        parser : OptionParser
                 Program option parser
        pgroup : OptionGroup
                 Parent option group
    '''
    from settings import OptionGroup
    levels=tuple(_log_level_val)
    group = OptionGroup(parser, "Logging", "Options to control the state of the logging module", id=__name__)
    group.add_option("-v", log_level=levels,    help="Set logging level application wide", default=3, dependent=False)
    group.add_option("",   log_file="",         help="Set file to log messages", gui=dict(filetype="save"), archive=True, dependent=False)
    group.add_option("",   log_config="",       help="File containing the configuration of the application logging", gui=dict(filetype="open"), dependent=False)
    group.add_option("",   enable_stderr=False, help="Enable logging to stderr along with --log-file", dependent=False)
    if pgroup is not None:
        pgroup.add_option_group(group)
    else:
        parser.add_option_group(group)
        
def default_logfile(rank=0, **extra):
    '''Generate a default crash report filename
    
    :Parameters:
        
        rank : int
               Identifier for process
        extra : dict
                Unused keyword arguments
    '''
    
    return "."+os.path.basename(sys.argv[0])+".crash_report.%d"%rank

def configure_logging(rank=0, log_level=3, log_file="", log_config="", enable_stderr=False, log_mode='w', **extra):
    '''Configure logging with use selected options

    .. sourcecode:: py
    
        >>> import core.app.tracing, logging
        >>> core.app.tracing.configure_logging(log_level=3)
    
    :Parameters:
    
        rank : int
               Identifier for process
        log_level : int
                    Level for logging application wide
        log_file : str
                   File path for logging messages
        log_config : str
                     File path for logging configuration
        enable_stderr : bool
                        Do redirect to stderr
        log_mode : str
                   Mode to open log file
        extra : dict
                Unused keyword arguments
    '''
    
    if log_level == 5:
        for name in Logger._log_map:
            if logging.getLogger(name).getEffectiveLevel() == logging.INFO:
                logging.getLogger(name).setLevel(logging.DEBUG)
    
    if log_config != "":
        logging.config.fileConfig(log_config)
    else:
        if rank != 0 and log_file != "":
            if log_file != "":
                base, ext = os.path.splitext(log_file)
                base += "_"
            else: base, ext = "", ".log"
            log_file = base+socket.gethostname()+"_"+str(rank)+ext
        handlers = []
        default_error_log = default_logfile(rank)
        
        try: 
            if log_file != "":
                logging.debug("Writing to log file: %s"%(log_file))
                try:
                    backupname = backup(log_file)
                except:
                    logging.warn("Unable to backup log file")
                else:
                    if backupname: logging.debug("Backing up log file to %s"%(backupname))
                h = logging.FileHandler(log_file, mode=log_mode)
                h.addFilter(ExceptionFilter())
                backupname = backup(default_error_log)
                if backupname: logging.debug("Backing up crash report to %s"%(backupname))
                handlers.append(logging.FileHandler(default_error_log, mode='w'))
                handlers.append(h)
                if enable_stderr: 
                    h = logging.StreamHandler()
                    h.addFilter(ExceptionFilter())
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
            try:
                backupname = backup(default_error_log)
                if backupname: logging.debug("Backing up crash report to %s"%(backupname))
                handlers.append(logging.FileHandler(default_error_log, mode='w'))
            except: pass
            else: ch.addFilter(ExceptionFilter())
            handlers.append(ch)
        try:    log_level_name = _log_level_val[log_level]
        except: log_level_name = log_level
        level = _log_level_map[log_level_name]
        logging.basicConfig(level=level)
        root = logging.getLogger()
        root.setLevel(level)
        while len(root.handlers) > 0: root.removeHandler(root.handlers[0])
        for ch in handlers:
            if not isinstance(ch, logging.FileHandler) and isinstance(ch, logging.StreamHandler) and supports_colors():
                ch.setFormatter(ColoredFormatter(_log_formats[log_level_name]))
            else: 
                ch.setFormatter(logging.Formatter(_log_formats[log_level_name]))
            root.addHandler(ch)
            ch.setLevel(level)
        root.setLevel(level)
    
    if rank == 0: print_import_warnings()
    
def configure_mp_logging(filename, level=logging.DEBUG, process_number=None, **extra):
    ''' Create a log file with given name for each process. It appends process number to end of 
    log file name.
    
    :Parameters:
        
        filename : str
                   Output log filename
        level : int
                Level for logging, default logging.DEBUG
        process_number : int
                         Process number passed by API
        extra : dict
                Unused keyword arguments
    '''
    
    if process_number is None: return
    filename, ext = os.path.splitext(filename)
    filename += "_%7d"%process_number+ext
    ch = logging.FileHandler(filename, mode='w')
    ch.setLevel(level)
    logging.getLogger().addHandler(ch)
    
def print_import_warnings():
    ''' Log failed imports being tracked as warnings
    '''
    
    for errormsg in _log_import_errors:
        logging.warn(errormsg, extra=dict(tofile=True))

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
        try: zf.write(filename, arcname=arcname)#, compress_type=zipfile.ZIP_STORED)
        except: pass
        else: os.unlink(filename)
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

def supports_colors():
    ''' Test whether terminal supports colors
    
    :Returns:
        
        flag : bool
               True if terminal supports colors
    '''
    
    try:
        import curses
        curses.setupterm()
        #window.getbkgd()
        return curses.tigetnum('colors') > 2
    except: return False

if not _loaded:
    _loaded = True
    
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
    logging.getLogger("arachnid").addHandler(h)
 
_RESET_SEQ = "\033[0m"
_COLOR_SEQ = "\033[1;%dm"
#BLACK, _RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
_RED=1
_BLUE=4
COLORS={'ERROR': _RED, 'WARN':_BLUE}   
class ColoredFormatter(logging.Formatter):
    ''' Highlight the levelname for errors in red
    '''

    def __init__(self, fmt=None, datefmt=None):
        ''' Initialize formatter object
        
        :Parameters:
            
            fmt : str
                  Message format string
            datefmt : str
                      Date format string
        '''
        
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        ''' Format the log record
        
        Highligh the levelname for errors in red
        
        :Parameters:
            
            record : str
                     Log record to format
        
        :Returns:
            
            text : str
                   Formatted text
        '''
        
        levelname = record.levelname
        if levelname in COLORS:
            levelname_color = _COLOR_SEQ % (30 + COLORS[levelname]) + levelname + _RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)

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
                  True if record does not contain an exception or
                  if record does not have `tofile` attribute.
        '''
        
        #print "here: ", record.exc_info
        #record.exc_info  = None
        return record.exc_info is None and not hasattr(record, 'tofile')
    

if logging._loggerClass != Logger: logging.setLoggerClass(Logger)
