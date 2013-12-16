'''
.. Created on Dec 7, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from pyui.ReferenceUI import Ui_Form
from util.qt4_loader import QtGui,QtCore,qtSlot, qtSignal
from util import BackgroundTask
import logging
import gzip
import urllib
import os
from urlparse import urlparse
import multiprocessing
from ..image import ndimage_file

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

class Widget(QtGui.QWidget): 
    ''' Display Controls for the LeginonUI
    '''
    
    taskFinished = qtSignal(object)
    
    def __init__(self, parent=None):
        "Initialize ReferenceUI widget"
        
        QtGui.QWidget.__init__(self, parent)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.task=None
        self.lastpath = str(QtCore.QDir.currentPath())
        
        
        self.emdbCannedModel = QtGui.QStandardItemModel(self)
        canned = [('Ribosome-70S', '2183', ':/icons/icons/ribosome_70S_32x32.png'),
                  ('Ribosome-50S', '1456', ':/icons/icons/ribosome_60S_32x32.png'),
                  ('Ribosome-30S', '5503', ':/icons/icons/ribosome30s_32x32.png'),
                  ('Ribosome-80S', '2275', ':/icons/icons/ribosome80s_32x32.png'),
                  ('Ribosome-60S', '1705', ':/icons/icons/ribosome_60S_32x32.png'),
                  ('Ribosome-40S', '1925', ':/icons/icons/ribosome_40S_32x32.png'),]
        for entry in canned:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(entry[2]), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item = QtGui.QStandardItem(icon, entry[0])
            item.setData(entry[1], QtCore.Qt.UserRole)
            self.emdbCannedModel.appendRow(item)
        self.ui.emdbCannedListView.setModel(self.emdbCannedModel)
        self.taskFinished.connect(self.onDownloadFromEMDBComplete)
        
        #self.ui.referenceLineEdit.setText(self.param['raw_reference'])
        #self.ui.referencePixelSizeDoubleSpinBox.setValue(self.param['curr_apix'])
        #self.openReference(self.param['raw_reference'])
    
    @qtSlot()
    def on_emdbDownloadPushButton_clicked(self):
        '''Called when the user clicks the download button
        '''
        
        num = self.ui.emdbNumberLineEdit.text()
        if num == "":
            QtGui.QMessageBox.warning(self, "Warning", "Empty Accession Number")
            return
        url="ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/emd_%s.map.gz"%(num, num)
        self.setEnabled(False)
        self.task=BackgroundTask.launch(self, download_gunzip_task, url, '.')
    
    def onDownloadFromEMDBComplete(self, local):
        ''' Called when the download and unzip is complete
        '''
        
        self.setEnabled(True)
        self.task.disconnect()
        if local == 1:
            QtGui.QMessageBox.critical(self, "Error", "Download failed - check accession number")
        elif local == 2:
            QtGui.QMessageBox.critical(self, "Error", "Unzip failed - check file")
        else:
            self.ui.referenceLineEdit.setText(os.path.abspath(local))
            self.ui.referenceTabWidget.setCurrentIndex(0)
        self.task=None
    
    @qtSlot(QtCore.QModelIndex)
    def on_emdbCannedListView_doubleClicked(self, index):
        ''' Called when the user clicks on the list
        '''
        
        num = index.data(QtCore.Qt.UserRole) #.toString()
        self.ui.emdbNumberLineEdit.setText(num)
    
    #@qtSlot(name='on_referenceLineEdit_editingFinished')
    def onReferenceEditChanged(self):
        '''
        '''
        
        text = str(self.ui.referenceLineEdit.text())
        self.openReference(text)
    
    @qtSlot()
    def on_referenceFilePushButton_clicked(self):
        '''Called when the user open reference button
        '''
        
        filename = QtGui.QFileDialog.getOpenFileName(self, self.tr("Open a reference volume"), self.lastpath)
        if isinstance(filename, tuple): filename = filename[0]
        if filename != "": 
            self.lastpath = os.path.dirname(filename)
            self.ui.referenceLineEdit.blockSignals(True)
            self.ui.referenceLineEdit.setText(self.openReference(filename))
            self.ui.referenceLineEdit.blockSignals(False)
    
    def openReference(self, filename):
        ''' Open a list of micrograph files
        '''
        
        if filename == "":
            self.param['raw_reference'] = str(filename)
        if filename != "" and not os.path.exists(filename):
            QtGui.QMessageBox.warning(self, "Warning", "File does not exist: %s"%filename)

        if filename != "" and os.path.exists(filename):
            img = ndimage_file.read_image(str(filename))
            
            if img.ndim == 3:
                self.ui.referenceWidthLabel.setText(str(img.shape[0]))
                self.ui.referenceHeightLabel.setText(str(img.shape[1]))
                self.ui.referenceDepthLabel.setText(str(img.shape[2]))
                self.param['raw_reference'] = str(filename)
            else:
                QtGui.QMessageBox.warning(self, "Warning", "File is not a volume: %s"%str(img.shape))
            if self.param['curr_apix'] == 0:
                header = ndimage_file.read_header(filename)
                self.ui.referencePixelSizeDoubleSpinBox.blockSignals(True)
                self.ui.referencePixelSizeDoubleSpinBox.setValue(header['apix'])
                self.ui.referencePixelSizeDoubleSpinBox.blockSignals(False)
                self.param['curr_apix'] = header['apix']

        return self.param['raw_reference']
    
    def registerPage(self, wizardPage):
        '''
        '''
        
        wizardPage.registerField(wizardPage.wizard().param("raw_reference_file*"), self.ui.referenceLineEdit)
                

def download_gunzip_task(urlpath, filepath):
    ''' Download and unzip gzipped file in a separate process
    
    :Parameters:
        
    urlpath : str
              Full URL to download the file from
    filepath : str
               Local path for filename
            
    :Returns:
    
    outputfile : str
                 Output filename
    '''
    
    def worker_callback(urlpath, filepath, qout):       
        try:
            filename=download(urlpath, filepath)
        except:
            qout.put(1)
            return
        try:
            filename=gunzip(filename)
        except:
            qout.put(2)
            return
        else:
            qout.put(filename)
    
    qout = multiprocessing.Queue()
    multiprocessing.Process(target=worker_callback, args=(urlpath, filepath, qout)).start()
    yield qout.get()

def gunzip(inputfile, outputfile=None):
    ''' Unzip a GZIPed file
    
    :Parameters:
    
    inputfile : str
                Input filename
    outputfile : str, optional
                 Output filename
                 
    :Returns:
    
    outputfile : str
                 Output filename
    '''
    
    if outputfile is None: 
        n = inputfile.rfind('.')
        outputfile=inputfile[:n]
    fin = gzip.open(inputfile, 'rb')
    fout = open(outputfile,"wb")
    fout.write(fin.read())
    fout.close()
    fin.close()
    return outputfile

def download(urlpath, filepath):
    '''Download the file at the given URL to the local filepath
    
    This function uses the urllib Python package to download file from to the remote URL
    to the local file path.
    
    :Parameters:
        
    urlpath : str
              Full URL to download the file from
    filepath : str
               Local path for filename
    
    :Returns:

    val : str
          Local filename
    '''
    
    filename = urllib.url2pathname(urlparse(urlpath)[2])
    filename = os.path.join(os.path.normpath(filepath), os.path.basename(filename))
    urllib.urlretrieve(urlpath, filename)
    return filename
