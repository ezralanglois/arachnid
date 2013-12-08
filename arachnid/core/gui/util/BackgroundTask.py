''' Launch tasks in the background

.. Created on Jan 29, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore, qtSignal

class TaskSignal(QtCore.QObject):
    '''
    '''
    taskFinished = qtSignal(object)
    taskStarted = qtSignal(object)
    taskUpdated = qtSignal(object)

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
        for val in self.functor(*self.args):
            self.signal.taskUpdated.emit(val)
        self.signal.taskFinished.emit(val)
    
    def connect_obj(self, parent, connect):
        '''
        '''
        
        for signal in ('taskFinished', 'taskStarted', 'taskUpdated'):
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

