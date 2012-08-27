''' Read/Write a table in the malibu prediction format (PRED)

This module reads from and writes to the malibu prediction format describes an entire machine learning experiment.

An example of the file:

.. container:: bottomnav, topic

    | #ET    wwillowboost | Testset | trnrf303.csv | 12 | 1 | B | 0
    | #CL    0,1
    | #ES    5852,0,2,1
    | #SP    prepwinser_15930:0001    0    -7.09427    0
    | #SP    prepwinser_15930:0002    0    -8.4979    0
    | #SP    prepwinser_15930:0003    0    -23.5452    0
    |         .
    |         .
    |         .
    | #XX

It supports the following attributes:
    
    - Extension: pred
    - Filter: Prediction (\*.pred)

.. Created on Apr 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..format_utility import ParseFormatError, convert
from ..factories import namedtuple_factory
from collections import namedtuple
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_experiment_size_class = namedtuple("ExperimentSize", "examples,bags,classes,labels")
_experiment_type_class = namedtuple("ExperimentType", "algorithm,validation,dataset,elapsed,runs,type,threshold")
_prediction_header = ["id", "select", "confidence", "threshold"]

class read_iterator(object):
    '''Malibu prediction format parsing iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.pred")
        #ET    wwillowboost | Testset | trnrf303.csv | 12 | 1 | B | 0
        #CL    0,1
        #ES    2,0,2,1
        #SP    prepwinser_15930:0001    0    -7.09427    0
        #SP    prepwinser_15930:0002    0    -8.4979    0
        
        >>> header = []
        >>> fin = open("data.pred", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "select", "confidence", "threshold"]
        >>> reader = read_iterator(fin, len(header), lastline)
        >>> map(factory, reader)
        [ BasicTuple("15930/1", 0, -7.09427, 0), BasicTuple("15930/2", 0, -8.4979, 0) ]
    
    :Parameters:
    
    fin : file stream
          Input file stream
    hlen : integer
           Length of the header
    lastline : string
               Last line read during header parsing, requires CSV parsing now
    numeric : boolean
              If true then convert string values to numeric
    columns : list
              List of columns to read otherwise None (all columns)
    extra : dict
            Unused keyword arguments
    '''
    
    __slots__=("fin", "hlen", "lastline", "numeric", "columns")
    
    def __init__(self, fin, hlen, lastline="", numeric=False, columns=None, **extra):
        "Create a read iterator"
        
        self.fin = fin
        self.hlen = hlen
        self.lastline=lastline
        self.numeric=numeric
        self.columns = columns
    
    def __iter__(self):
        '''Get iterator for class
        
        This class defines its own iterator.
        
        :Returns:
        
        val : iterator
              Self
        '''
        
        return self
    
    def next(self):
        '''Go to the next non-comment line
        
        This method skips to next non-comment line, parses the line into a list of values
        and returns those values. It raises StopIteration when it is finished.
        
        :Returns:
        
        val : list
              List of values parsed from current line of the file
        '''
        
        if self.lastline == "":
            while True:
                line = self.fin.readline()
                if line == "": 
                    self.fin.close()
                    raise StopIteration
                line = line.strip()
                if line == "" or line[0:3] != '#SP' or line == "#SP\t//": 
                    continue
                break
        else:
            line = self.lastline
            self.lastline = ""
        vals = line.split('\t')[1:]
        _logger.debug("line:"+str(self.hlen)+" == "+str(len(vals)))
        if self.hlen != len(vals): raise ParseFormatError, "Header length does not match values: "+str(self.hlen)+" != "+str(len(vals))+" --> "+str(vals)
        if self.columns is not None: vals = vals[self.columns]
        if self.numeric: return [convert(v) for v in vals]
        return vals

def read_header(filename, header=[], factory=namedtuple_factory, description={}, **extra):
    '''Parses the multi-line header of a malibu prediction file
    
    The description dictionary will contain the following keys:
    
        - experiment_size: a namedtuple with the fields: examples,bags,classes,labels
        - experiment_type: a namedtuple with the fields: algorithm,validation,dataset,elapsed,runs,type,threshold
        - class_names: a list of strings
        - Additional comments
        
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.pred")
        #ET    wwillowboost | Testset | trnrf303.csv | 12 | 1 | B | 0
        #CL    0,1
        #ES    2,0,2,1
        #SP    prepwinser_15930:0001    0    -7.09427    0
        #SP    prepwinser_15930:0002    0    -8.4979    0
        
        >>> header = []
        >>> fin = open("data.pred", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "select", "confidence", "threshold"] ]
    
    :Parameters:
    
    filename : string or stream
               Input filename or stream
    header : list
             List of strings overriding parsed header
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    description : dict
                  Description from the header of the prediction file
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : container
          Container with the given header values
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    try:
        while True:
            line = fin.readline()
            if line == "": break
            line = line.strip()
            if line == "": continue
            if line.find('\t') == -1: raise ParseFormatError, "Could not parse prediction HEADER - no tab"
            tag, val = line.split('\t', 1)
            if tag == '#ES':
                try:
                    description["experiment_size"] = _experiment_size_class._make(val.split(','))
                    _logger.debug("experiment_size:"+str(description["experiment_size"]))
                except:
                    _logger.error("Error parsing experiment size: "+val+" -- "+str(val.split(',')))
                    raise
            else:
                if tag == '#CM':
                    try:
                        k, v = val.split('=', 1)
                        description[k.strip()] = v.strip()
                    except: pass
                elif tag == '#CL':  description["class_names"] = val.split(',')
                elif tag == '#ET':
                    try:
                        description["experiment_type"] = _experiment_type_class._make(val.split(' | '))
                        _logger.debug("experiment_type:"+str(description["experiment_type"]))
                    except:
                        _logger.error("Error parsing experiment type: "+val+" -- "+str(val.split(' | '))+" | "+str(_experiment_type_class._fields))
                        raise
                continue
            header.extend(_prediction_header)
            return factory.create(_prediction_header, description=description, **extra), header, ""
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Error reading prediction header")
        fin.close()
        raise
    else:
        fin.close()
    raise ParseFormatError, "Cannot parse header of prediction file - end of document"

def reader(filename, header=[], lastline="", **extra):
    '''Creates a malibu prediction read iterator
    
    .. sourcecode:: py
        
        >>> import os
        >>> os.system("more data.pred")
        #ET    wwillowboost | Testset | trnrf303.csv | 12 | 1 | B | 0
        #CL    0,1
        #ES    2,0,2,1
        #SP    prepwinser_15930:0001    0    -7.09427    0
        #SP    prepwinser_15930:0002    0    -8.4979    0
        
        >>> header = []
        >>> fin = open("data.pred", 'r')
        >>> factory, lastline = read_header(fin, header)
        >>> header
        ["id", "select", "confidence", "threshold"]
        >>> reader = read_iterator(fin, len(header), lastline)
        >>> map(factory, reader)
        [ BasicTuple("15930/1", 0, -7.09427, 0), BasicTuple("15930/2", 0, -8.4979, 0) ]
    
    :Parameters:
    
    filename : string or stream
               Input filename or input stream
    header : list
             List of strings overriding parsed header
    lastline : string
              Last line read by header parser, first line to parse
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    val : iterator
          malibu prediction read iterator
    '''
    
    fin = open(filename, 'r') if isinstance(filename, str) else filename
    return read_iterator(fin, len(header), lastline, **extra)

############################################################################################################
# Write format                                                                                             #
############################################################################################################

def create_default_header(count, class_names=[], experiment_size=[], experiment_type=[], **extra):
    '''Create a default malibu prediction header
    
    This function creates a default header description by filling the given lists, if 
    they do not already have the appropriate values.
    
    :Parameters:
    
    count : int
            Number of values
    class_names : list
                  Output list of string class names
    experiment_size : list
                      Output list of integers describing an experiment size
    experiment_type : list
                      Output list of string values describing the experiment type
    extra : dict
            Unused keyword arguments
    '''
    
    if len(class_names) == 0:
        class_count = 2 if len(experiment_size) < 3 else int(experiment_size[2])
        if class_count == 0: class_count = 2
        for i in xrange(class_count): class_names.append(str(i))
    else: class_count = len(class_names)
    
    if len(experiment_type) != 7:
        type = 'B' if class_count == 2 else 'M'
        if len(experiment_type) < 1: experiment_type.append("AutoPart")
        if len(experiment_type) < 2: experiment_type.append("Holdout")
        if len(experiment_type) < 3: experiment_type.append("Unknown")
        if len(experiment_type) < 4: experiment_type.append("0")
        if len(experiment_type) < 5: experiment_type.append("1")
        if len(experiment_type) < 6: experiment_type.append(type)
        if len(experiment_type) < 7: experiment_type.append("0")
    
    if len(experiment_size) != 4:
        if len(experiment_size) < 1: experiment_size.append(str(count))
        if len(experiment_size) < 2: experiment_size.append('0')
        if len(experiment_size) < 3: experiment_size.append(str(class_count))
        if len(experiment_size) < 4: experiment_size.append("1")

def write_header(fout, values, factory=namedtuple_factory, class_names=[], experiment_size=[], experiment_type=[], **extra):
    '''Write a malibu prediction header
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,confidence,threshold")
        >>> values = [ BasicTuple("15930/1", 0, -7.09427, 0), BasicTuple("15930/2", 0, -8.4979, 0) ]
        >>> write_header("data.pred", values)
        
        >>> import os
        >>> os.system("more data.pred")
        #ET    AutoPart | Holdout | Unknown | 0 | 1 | B | 0
        #CL    0,1
        #ES    2,0,2,1
    
    :Parameters:
    
    fout : stream
           Output stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    class_names : list
                  List of string class names
    experiment_size : list
                      List of integers describing an experiment size
    experiment_type : list
                      List of string values describing the experiment type
    extra : dict
            Unused keyword arguments
    '''
    
    create_default_header(len(values), class_names, experiment_size, experiment_type, **extra)
    fout.write("#ET\t"+" | ".join(experiment_type)+"\n")
    fout.write("#CL\t"+",".join(class_names)+"\n")
    fout.write("#ES\t"+",".join(experiment_size)+"\n")

def write_values(fout, values, factory=namedtuple_factory, header=_prediction_header, **extra):
    '''Write malibu prediction values
    
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,confidence,threshold")
        >>> values = [ BasicTuple("15930/1", 0, -7.09427, 0), BasicTuple("15930/2", 0, -8.4979, 0) ]
        >>> write_values("data.pred", values)
        
        >>> import os
        >>> os.system("more data.pred")
        #SP    prepwinser_15930:0001    0    -7.09427    0
        #SP    prepwinser_15930:0002    0    -8.4979    0
    
    :Parameters:
    
    fout : stream
           Output stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    header : list
             List of string describing the header
    extra : dict
            Unused keyword arguments
    '''
    
    _logger.debug("header-1: %s"%str(header))
    header = factory.get_header(values, header=_prediction_header, offset=True, **extra)
    _logger.debug("header-2: %s"%str(header))
    for v in values:
        vals = factory.get_values(v, header, **extra)
        fout.write("#SP\t"+"\t".join(vals)+"\n")
    fout.write('#XX\n\n')

def write(filename, values, factory=namedtuple_factory, **extra):
    '''Write a malibu prediction file
    
    .. sourcecode:: py
        
        >>> BasicTuple = namedtuple("BasicTuple", "id,select,confidence,threshold")
        >>> values = [ BasicTuple("15930/1", 0, -7.09427, 0), BasicTuple("15930/2", 0, -8.4979, 0) ]
        >>> write("data.pred", values)
        
        >>> import os
        >>> os.system("more data.pred")
        #ET    AutoPart | Holdout | Unknown | 0 | 1 | B | 0
        #CL    0,1
        #ES    2,0,2,1
        #SP    prepwinser_15930:0001    0    -7.09427    0
        #SP    prepwinser_15930:0002    0    -8.4979    0
    
    :Parameters:
    
    filename : string or stream
               Output filename or stream
    values : container
             Value container such as a list or an ndarray
    factory : Factory
              Class or module that creates the container for the values returned by the parser
    extra : dict
            Unused keyword arguments
    '''
    
    fout = open(filename, 'w') if isinstance(filename, str) else filename
    write_header(fout, values, factory, **extra)
    write_values(fout, values, factory, **extra)
    if isinstance(filename, str): fout.close()


############################################################################################################
# Extension and Filters                                                                                    #
############################################################################################################

def extension():
    '''Get extension of the malibu prediction format
    
    :Returns:
    
    val : string
          File extension - pred
    '''
    
    return "pred"

def filter():
    '''Get filter of malibu prediction format
    
    :Returns:
    
    val : string
          File filter - Prediction (\*.pred)
    '''
    
    return "Prediction (*.pred)"

        

