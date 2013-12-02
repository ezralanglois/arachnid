''' A monitor for a running process

.. Created on Nov 1, 2013
.. codeauthor:: robertlanglois
'''
from util.qt4_loader import QtCore, QtGui, qtSignal, qtSlot
from pyui.Monitor import Ui_Form
import logging, os, psutil

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget):
    '''
    '''
    
    runProgram = qtSignal()
    fireProgress = qtSignal(int)
    fireMaximum = qtSignal(int)
    
    def __init__(self, parent=None):
        '''
        '''
        
        QtGui.QWidget.__init__(self, parent)
        
        self.ui = Ui_Form()
        
        self.ui.jobUpdateTimer = QtCore.QTimer(self)
        self.ui.jobUpdateTimer.setInterval(500)
        self.ui.jobUpdateTimer.setSingleShot(False)
        
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.runProgram)
        self.ui.jobUpdateTimer.timeout.connect(self.on_jobUpdateTimer_timeout)
        
        self.fireProgress.connect(self.ui.jobProgressBar.setValue)
        self.fireMaximum.connect(self.ui.jobProgressBar.setMaximum)
        self.ui.jobProgressBar.setMinimum(0)
        
        #self.text_cursor = QtGui.QTextCursor(self.ui.logTextEdit.document())
        self.current_pid = None
        self.fin = None
        self.log_file = None
        self.created = None
    
    def setLogFile(self, filename):
        '''
        '''
        
        self.log_file = filename
        if os.path.exists(self.log_file):
            lines = self.readLogFile()
            if len(lines) == 0: return
            if self.isComplete(lines): return
            #created = self.parsePID(lines, 'Completed')
            self.current_pid = self.parsePID(lines)
            if self.current_pid is not None:
                created = self.parsePID(lines, 'Created:')
                if self.isRunning(created):
                    self.current_pid = None
                    self.created = None
                    self.fin = None
                    self.ui.pushButton.setChecked(QtCore.Qt.Checked)
                else:
                    self.current_pid = None
                    self.fin = None
 
    @qtSlot(bool)
    def on_pushButton_toggled(self, checked=False):
        '''
        '''
        
        self.ui.jobProgressBar.setValue(0)
        if checked:
            self.current_pid = None
            self.created = None
            self.fin = None
            self.ui.logTextEdit.setPlainText("")
            self.ui.jobUpdateTimer.setInterval(5000)
            self.ui.jobUpdateTimer.start()
        else:
            self.ui.jobUpdateTimer.stop()
            self.ui.jobProgressBar.setMaximum(0)
            if self.fin is not None:
                self.fin.close()
                self.fin = None
            self.current_pid = None
            self.created = None
            self.fin = None
    
    #@qtSlot()
    def on_jobUpdateTimer_timeout(self):
        '''
        '''
        
        if self.ui.jobUpdateTimer.interval() != 500:
            self.ui.jobUpdateTimer.setInterval(500)
        
        if self.current_pid is None:
            if not os.path.exists(self.log_file):
                self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
                return
        
        lines = self.readLogFile() # handel missing newline at end!
        if len(lines) == 0: return
        
        if self.current_pid is None:
            self.current_pid = self.parsePID(lines)
            if self.current_pid is not None:
                self.created = self.parsePID(lines, 'Created:')
            if not self.isRunning(self.created):
                self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
                return
        
        text_cursor = self.ui.logTextEdit.textCursor()
        text_cursor.movePosition(QtGui.QTextCursor.Start)
        for line in lines:
            text_cursor.insertText(line)
        self.ui.logTextEdit.setTextCursor(text_cursor)
        '''
        self.text_cursor.movePosition(QtGui.QTextCursor.Start)
        for line in lines:
            self.text_cursor.insertText(line)
        '''
        self.updateProgress(lines)
        if self.isComplete(lines):
            self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
    
    def isRunning(self, created):
        '''
        '''
        
        if self.current_pid is None: return False
        try:
            p = psutil.Process(self.current_pid)
        except psutil.NoSuchProcess: 
            return False
        return created == int(p.create_time)
    
    def isComplete(self, lines):
        '''
        '''
        
        for i in xrange(len(lines)):
            idx = lines[i].find('Completed')
            if idx != -1: return True
        return None
    
    def parsePID(self, lines, tag='PID:'):
        '''
        '''
        
        for i in xrange(len(lines)-1, -1, -1):
            idx = lines[i].find(tag)
            if idx != -1:
                line = lines[i][idx+len(tag):]
                return int(line)
        return None
        
    def updateProgress(self, lines, tag='Finished: '):
        '''
        '''
        
        progress, maximum = None, None
        for i in xrange(len(lines)):
            line = lines[i]
            idx = line.find(tag)
            if idx != -1:
                line = line[idx+len(tag):]
                idx = line.find(' -')
                if idx != -1: line = line[:idx]
                progress, maximum = tuple([int(v) for v in line.split(',')])
                print 'progress:', progress, maximum, self.ui.jobProgressBar.maximum()
                if self.ui.jobProgressBar.maximum() != (maximum+1):
                    print 'progress2:', progress, maximum, self.ui.jobProgressBar.maximum()
                    self.fireMaximum.emit(maximum+1)
                    self.fireProgress.emit(1)
                self.fireProgress.emit(progress)
                QtGui.QApplication.processEvents()
                return
    
    def readLogFile(self):
        '''
        '''
        
        if self.fin is None:
            if not os.path.exists(self.log_file): 
                return []
            try:
                self.fin = open(self.log_file, 'rb')
            except: 
                self.fin = None
                return []
        try:
            lines = self.fin.readlines(1048576)
            lines.reverse()
            return lines
        except:
            return []
        
    
        
        
