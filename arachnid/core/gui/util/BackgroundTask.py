''' Launch tasks in the background

.. Created on Jan 29, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..util.qt4_loader import QtCore

class BackgroundTask(QtCore.QRunnable):
    def __init__(self, parent, functor, *args):
        QtCore.QRunnable.__init__(self)
        self.functor=functor
        self.args=args
        self.signal = QtCore.QObject()
        self.connect_obj(parent, QtCore.QObject.connect)
        self._parent=parent
    
    def run(self):
        self.signal.emit(QtCore.SIGNAL('taskStarted(PyQt_PyObject)'), self.args)
        for val in self.functor(*self.args):
            self.signal.emit(QtCore.SIGNAL('taskUpdated(PyQt_PyObject)'), val)
        self.signal.emit(QtCore.SIGNAL('taskFinished(PyQt_PyObject)'), val)
    
    def connect_obj(self, parent, connect):
        '''
        '''
        
        connect(self.signal, QtCore.SIGNAL('taskStarted(PyQt_PyObject)'), parent, QtCore.SIGNAL('taskStarted(PyQt_PyObject)'))
        connect(self.signal, QtCore.SIGNAL('taskUpdated(PyQt_PyObject)'), parent, QtCore.SIGNAL('taskUpdated(PyQt_PyObject)'))
        connect(self.signal, QtCore.SIGNAL('taskFinished(PyQt_PyObject)'), parent, QtCore.SIGNAL('taskFinished(PyQt_PyObject)'))
    
    def disconnect(self, val=None):
        print "disconnecing"
        self.connect_obj(self._parent, QtCore.QObject.disconnect)

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