''' Launch tasks in the background

.. Created on Jan 29, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore, qtSignal
import multiprocessing
import sys
import logging


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
    
class TaskException(Exception):
    
    def __init__(self, type=None, value=None):
        if type is None:
            self.exc_type, self.exc_value = sys.exc_info()[:2]
    def __str__(self):
        return repr(self.exc_value)

class TaskSignal(QtCore.QObject):
    '''
    '''
    taskFinished = qtSignal(object)
    taskStarted = qtSignal(object)
    taskUpdated = qtSignal(object)
    taskError = qtSignal(object)

class BackgroundTask(QtCore.QRunnable):
    def __init__(self, parent, functor, *args):
        QtCore.QRunnable.__init__(self)
        self.functor=functor
        self.args=args
        self.signal = TaskSignal()
        self.connect_obj(parent, 'connect')
        self._parent=parent
    
    def run(self):
        self.signal.taskStarted.emit(self.args)
        try:
            for val in self.functor(*self.args):
                self.signal.taskUpdated.emit(val)
        except TaskException, exp:
            self.signal.taskError.emit(exp)
            _logger.exception("task failed")
        except:
            self.signal.taskError.emit(sys.exc_info()[:2])
            _logger.exception("unknown task failed")
        else:
            self.signal.taskFinished.emit(val)
        self.connect_obj(self._parent, 'disconnect')
    
    def connect_obj(self, parent, connect):
        '''
        '''
        
        for signal in ('taskFinished', 'taskStarted', 'taskUpdated', 'taskError'):
            if not hasattr(parent, signal): continue
            sig = getattr(self.signal, signal)
            psig = getattr(parent, signal)
            getattr(sig, connect)(psig)
            
    def disconnect(self, val=None):
        self.connect_obj(self._parent, 'disconnect')

def launch(parent, functor, *args):
    '''Launch a task into the background using QT Threads
    
    :Parameters:
    
    :Returns:
    
    task : BackgroundTask
           Background task object
    '''
    
    task = BackgroundTask(parent, functor, *args)
    QtCore.QThreadPool.globalInstance().start(task)
    return task

def launch_mp(parent, functor, *args):
    '''Launch a task into the background using QT Threads
    
    :Parameters:
    
    :Returns:
    
    task : BackgroundTask
           Background task object
    '''
    
    return launch(parent, _mp_wrapper(functor), *args)

def _mp_wrapper(functor):
    '''
    '''
    
    def worker(qout, *args):
        try:
            for val in functor(*args):
                qout.put(val)
        except: 
            qout.put(TaskException())
            _logger.exception("worker failed")
        else:
            qout.put(None)
    
    def wrapper(*args):
        qout = multiprocessing.Queue()
        multiprocessing.Process(target=worker, args=(qout, )+args).start()
        while True:
            val = qout.get()
            if isinstance(val, TaskException): raise val
            if val is None: break
            yield val
    return wrapper
    

