''' A monitor for a running process


- Handel errors: Add crash report button, display on error in Details

.. Created on Nov 1, 2013
.. codeauthor:: robertlanglois
'''
from util.qt4_loader import QtCore, QtGui, qtSignal, qtSlot
from pyui.Monitor import Ui_Form
from ..app import tracing
import logging, os, psutil
import multiprocessing

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget):
    '''
    '''
    
    runProgram = qtSignal()
    monitorProgram = qtSignal()
    fireProgress = qtSignal(int)
    fireMaximum = qtSignal(int)
    captureScreen = qtSignal(int)
    programCompleted = qtSignal(str)
    programStarted = qtSignal(str)
    
    def __init__(self, parent=None, helpDialog=None):
        '''
        '''
        
        QtGui.QWidget.__init__(self, parent)
        
        self.ui = Ui_Form()
        
        self.timer_interval=500
        self.ui.jobUpdateTimer = QtCore.QTimer(self)
        self.ui.jobUpdateTimer.setInterval(self.timer_interval)
        self.ui.jobUpdateTimer.setSingleShot(False)
        self.helpDialog=helpDialog
        self.last_offset=0
        self.total_running=0
        
        self.ui.setupUi(self)
        #self.ui.pushButton.clicked.connect(self.runProgram)
        self.ui.jobUpdateTimer.timeout.connect(self.on_jobUpdateTimer_timeout)
        self.ui.jobProgressBar.setMinimum(0)
        self.ui.jobListView.setModel(QtGui.QStandardItemModel(self))
        self.job_status_icons=[QtGui.QIcon(f) for f in [':/mini/mini/clock.png', ':/mini/mini/arrow_refresh.png', ':/mini/mini/tick.png', ':/mini/mini/cross.png']]
        self.ui.crashReportToolButton.setEnabled(False)
        #self.text_cursor = QtGui.QTextCursor(self.ui.logTextEdit.document())
        self.current_pid = None
        self.fin = None
        self.log_file = None
        self.created = None
        self.log_text=""
        self.workflowProcess=None
    
    @qtSlot()
    def on_monitorInformationToolButton_clicked(self):
        '''
        '''
        
        if self.helpDialog is not None:
            self.helpDialog.setHTML(self.ui.pushButton.toolTip())
            self.helpDialog.show()
        else:
            QtGui.QToolTip.showText(self.ui.pushButton.mapToGlobal(QtCore.QPoint(0,0)), self.ui.pushButton.toolTip())
    
    def workflow(self):
        '''
        '''
        
        flow = []
        model = self.ui.jobListView.model()
        for i in xrange(model.rowCount()):
            flow.append( model.data(model.index(i, 0), QtCore.Qt.UserRole) )
        return flow
    
    def saveState(self):
        '''
        '''
        
        # save workflow in INI file
        
        for prog in self.workflow():
            prog.write_config()
    
    def setWorkflow(self, workflow):
        '''
        '''
        
        if not hasattr(workflow, '__iter__'): workflow=[workflow]
        model = self.ui.jobListView.model()
        model.clear()
        mode='w'
        for mod in workflow:
            if self.log_file:
                mod.values.log_file = self.log_file
            mod.values.log_mode = mode
            item = QtGui.QStandardItem(self.job_status_icons[0], mod.name())
            item.setData(mod, QtCore.Qt.UserRole)
            model.appendRow(item)
            if mode != 'a': mode='a'
    
    def model(self):
        '''
        '''
        
        return self.ui.jobListView.model()
    
    def setLogFile(self, filename):
        '''
        '''
        
        self.log_file = filename
        if not os.path.exists(self.log_file): return
        
        lines = self.readLogFile(True)
        if len(lines) == 0: return
        self.current_pid = self.parsePID(lines)
        if self.current_pid is not None:
            created = self.parsePID(lines, 'Created:')
            if self.isRunning(created):
                self.current_pid = None
                self.created = None
                self.fin = None
                self.monitorProgram.emit()
                self.ui.pushButton.setChecked(QtCore.Qt.Checked)
                model = self.ui.jobListView.model()
                model.item(0).setIcon(self.job_status_icons[1])
                self.programStarted.emit(model.item(0).text())
                self.ui.crashReportToolButton.setEnabled(False)
            else:
                self.testCompletion(lines)
                self.current_pid = None
                self.fin = None
    
    def testCompletion(self, lines, offset=0):
        '''
        '''
        
        model = self.ui.jobListView.model()
        if model.rowCount() == 0: return
        if self.isComplete(lines):
            model.item(0).setIcon(self.job_status_icons[2])
            self.ui.crashReportToolButton.setEnabled(False)
        else:
            model.item(0).setIcon(self.job_status_icons[3])
            self.ui.crashReportToolButton.setEnabled(True)
    
    @qtSlot()
    def on_crashReportToolButton_clicked(self):
        '''
        '''
        
        if self.ui.crashReportToolButton.isChecked():
            self.log_text = self.ui.logTextEdit.toPlainText()
            text = ""
            try:
                text = "".join(open(tracing.default_logfile(), 'r').readlines())
            except:
                _logger.error("Failed to read crash report")
            self.ui.logTextEdit.setPlainText(text)
        else:
            self.ui.logTextEdit.setPlainText(self.log_text)
    
    @qtSlot(bool)
    def on_pushButton_toggled(self, checked=False):
        '''
        '''
        
        self.ui.jobProgressBar.setValue(0)
        if checked:
            self.run_program()
            self.current_pid = None
            self.created = None
            self.fin = None
            self.ui.logTextEdit.setPlainText("")
            self.ui.jobUpdateTimer.setInterval(2000)
            self.ui.jobUpdateTimer.start()
        else:
            self.ui.jobUpdateTimer.stop()
            self.ui.jobProgressBar.setMaximum(1)
            if self.fin is not None:
                try: self.fin.close()
                except: pass
                self.fin = None
            self.current_pid = None
            self.created = None
            self.fin = None
    
    def run_program(self):
        '''
        '''
        
        if self.isRunning():
            _logger.error("Already running!")
            return
        
        def _run_worker(workflow):
            _logger.info("Workflow started")
            for prog in workflow:
                _logger.info("Running "+str(prog.name()))
                try:
                    prog.check_options_validity()
                except:
                    _logger.exception("Invalid option")
                    break
                try:
                    prog.launch()
                except: 
                    break
            _logger.info("Workflow ended")
        self.workflowProcess=multiprocessing.Process(target=_run_worker, args=(self.workflow(), ))
        self.workflowProcess.start()
        self.runProgram.emit()
        model = self.ui.jobListView.model()
        for i in xrange(1, model.rowCount()):
            model.item(i).setIcon(self.job_status_icons[0])
        model.item(0).setIcon(self.job_status_icons[1])
        self.programStarted.emit(model.item(0).text())
        self.ui.crashReportToolButton.setEnabled(False)
        self.captureScreen.emit(1)
    
    #@qtSlot()
    def on_jobUpdateTimer_timeout(self):
        '''
        '''
        
        if self.ui.jobUpdateTimer.interval() != self.timer_interval:
            self.ui.jobUpdateTimer.setInterval(self.timer_interval)
        
        if self.log_file is None:
            self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
            return
            
        if self.current_pid is None:
            if not os.path.exists(self.log_file):
                self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
                return
        
        lines = self.readLogFile() # handel missing newline at end!

        if len(lines) == 0: 
            return
        
        if self.current_pid is None:
            self.current_pid = self.parsePID(lines)
            if self.current_pid is not None:
                self.created = self.parsePID(lines, 'Created:')
            if not self.isRunning(self.created):
                self.testCompletion(lines)
                self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
                return
        
        text_cursor = self.ui.logTextEdit.textCursor()
        text_cursor.movePosition(QtGui.QTextCursor.Start)
        for line in lines:
            text_cursor.insertText(line)
        self.ui.logTextEdit.setTextCursor(text_cursor)
        
        self.updateListIcon(lines)
        '''
        self.text_cursor.movePosition(QtGui.QTextCursor.Start)
        for line in lines:
            self.text_cursor.insertText(line)
        '''
        self.updateProgress(lines)
        self.updateRunning(lines)
        
        if self.parseName(lines, 'Workflow ended') is not None:
            no_error = self.total_running == 0
            self.updateListIconFromOffset(None, not no_error)
            self.ui.pushButton.setChecked(QtCore.Qt.Unchecked)
      
    def updateListIcon(self, lines):
        '''
        '''
        
        program = self.parseName(lines)
        if program is None: return
        program = program.strip()
        model = self.ui.jobListView.model()
        for offset in xrange(model.rowCount()):
            if str(model.item(offset).data(QtCore.Qt.UserRole).id()) == program:
                break
        if offset == model.rowCount(): 
            _logger.error("Could not find ID: %s"%program)
            return
        self.updateListIconFromOffset(offset)
    
    def updateListIconFromOffset(self, offset=None, stopped=None):
        '''
        '''
        
        model = self.ui.jobListView.model()
        if offset is None: offset=self.last_offset
        for i in xrange(offset):
            if model.item(i).icon() != self.job_status_icons[2]:
                self.programCompleted.emit(model.item(i).text())
            model.item(i).setIcon(self.job_status_icons[2])
        
        self.last_offset=offset
        if stopped is None:
            if model.item(offset).icon() != self.job_status_icons[2]:
                self.programCompleted.emit(model.item(offset).text())
            model.item(offset).setIcon(self.job_status_icons[2])
        elif not stopped:
            model.item(offset).setIcon(self.job_status_icons[3])
        else:
            model.item(offset).setIcon(self.job_status_icons[1])
            self.programStarted.emit(model.item(offset).text())
        
        
    def isRunning(self, created=None):
        '''
        '''
        if created is None:
            lines = self.readLogFile(True)
            self.current_pid = self.parsePID(lines)
            created = self.parsePID(lines, 'Created:')
        
        if self.workflowProcess is not None:
            if not self.workflowProcess.is_alive(): 
                self.workflowProcess.terminate()
        if self.current_pid is None: return False
        try:
            p = psutil.Process(self.current_pid)
        except psutil.NoSuchProcess: 
            return False
        return created == int(p.create_time())
    
    def isComplete(self, lines):
        '''
        '''
        
        for i in xrange(len(lines)):
            idx = lines[i].find('Completed')
            if idx != -1: return True
        return None
    
    def parseName(self, lines, tag='Program:'):
        '''
        '''
        
        for i in xrange(len(lines)):
            idx = lines[i].find(tag)
            if idx != -1:
                line = lines[i][idx+len(tag):]
                return line
        return None
    
    def parseNames(self, lines, tag='Program:'):
        '''
        '''
        
        names = []
        for i in xrange(len(lines)):
            idx = lines[i].find(tag)
            if idx != -1:
                line = lines[i][idx+len(tag):]
                names.append(line.strip())
        return names
    
    def parsePID(self, lines, tag='PID:'):
        '''
        '''
        
        val = self.parseName(lines, tag)
        if val is not None: return int(val)
        return None
    
    def updateRunning(self, lines):
        '''
        '''
        
        self.total_running += len(self.parseNames(lines))
        self.total_running -= len(self.parseNames(lines, 'Completed'))
        
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
                if self.ui.jobProgressBar.maximum() != (maximum+1):
                    self.ui.jobProgressBar.setMaximum(maximum+1)
                self.ui.jobProgressBar.setValue(progress+1)
                #QtGui.QApplication.processEvents()
                return
    
    def readLogFile(self, once=False):
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
            if once:
                try:self.fin.close()
                except: pass
                self.fin = None
            return lines
        except:
            return []
        
    
        
        
