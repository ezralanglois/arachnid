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
    captureScreen = qtSignal(int)
    
    def __init__(self, parent=None, helpDialog=None):
        "Initialize ReferenceUI widget"
        
        QtGui.QWidget.__init__(self, parent)
        
        # Build window
        _logger.info("\rBuilding main window ...")
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.task=None
        self.lastpath = str(QtCore.QDir.currentPath())
        self.helpDialog=helpDialog
        self.ui.referenceTabWidget.currentChanged.connect(lambda x: self.captureScreen.emit(x+1))
        
        self.emdbCannedModel = QtGui.QStandardItemModel(self)
        canned = [('Ribosome-40S', '1346', ':/references/reference/1346/emdb_1346_pix32.png', 
"""The eukaryotic translation initiation factors eIF1 and eIF1A induce an open conformation of the 40S ribosome.
Passmore LA, Schmeing TM, Maag D, Applefield DJ, Acker MG, Algire MA, Lorsch JR, Ramakrishnan V
MOLECULAR CELL (2007) 26, pp. 41-50"""),
                  ('Ribosome-60S', '1705', ':/references/reference/1705/emdb_1705_pix32.png', 
"""Mechanism of eIF6-mediated inhibition of ribosomal subunit joining.
Gartmann M, Blau M, Armache JP, Mielke T, Topf M, Beckmann R
J.BIOL.CHEM. (2010) 285, pp. 14848-14851"""),
                  ('Ribosome-80S', '2275', ':/references/reference/2275/emdb_2275_pix32.png',
"""Ribosome structures to near-atomic resolution from thirty thousand cryo-EM particles.
Bai XC, Fernandez IS, McMullan G, Scheres SH
ELIFE (2013) 2, pp. e00461-e00461"""),
                  ('ATP synthase', '5335', ':/references/reference/5335/emd_5335_pix32.png',
"""Subnanometre-resolution structure of the intact Thermus thermophilus H+-driven ATP synthase.
Lau WC, Rubinstein JL
NATURE (2012) 481, pp. 214-218"""),
                  ('Ribosome-70S', '5360', ':/references/reference/5360/emdb_5360_pix32.png',
"""Structural characterization of mRNA-tRNA translocation intermediates.
Agirrezabala X, Liao HY, Schreiner E, Fu J, Ortiz-Meoz RF, Schulten K, Green R, Frank J
PROC.NAT.ACAD.SCI.USA (2012) 109, pp. 6094-6099"""),
                  ('Ribosome-30S', '5503', ':/references/reference/5503/emdb_5503_pix512.png',
"""Dissecting the in vivo assembly of the 30S ribosomal subunit reveals the role of RimM and general features of the assembly process.
Guo Q, Goto S, Chen Y, Feng B, Xu Y, Muto A, Himeno H, Deng H, Lei J, Gao N
NUCLEIC ACIDS RES. (2013) 41, pp. 2609-2620"""),
                  ('Ribosome-50S', '5787', ':/references/reference/5787/emdb_5787_pix32.png',
"""Functional domains of the 50S subunit mature late in the assembly process.
Jomaa A, Jain N, Davis JH, Williamson JR, Britton RA, Ortega J
NUCLEIC ACIDS RES. (2013)"""),]
        for entry in canned:
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(entry[2]), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item = QtGui.QStandardItem(icon, entry[0])
            item.setData(entry[1], QtCore.Qt.UserRole)
            item.setData(entry[3], QtCore.Qt.UserRole+1)
            self.emdbCannedModel.appendRow(item)
        self.ui.emdbCannedListView.setModel(self.emdbCannedModel)
        self.taskFinished.connect(self.onDownloadFromEMDBComplete)
        self.ui.referenceLineEdit.editingFinished.connect(self.onReferenceEditChanged)
        self.ui.referenceTabWidget.setCurrentIndex(0)
        self.ui.referenceTabWidget.setCurrentIndex(1)
        
    
    @qtSlot()
    def on_downloadInformationToolButton_clicked(self):
        '''
        '''
        
        if self.helpDialog is not None:
            self.helpDialog.setHTML(self.ui.emdbDownloadPushButton.toolTip())
            self.helpDialog.show()
        else:
            QtGui.QToolTip.showText(self.ui.emdbDownloadPushButton.mapToGlobal(QtCore.QPoint(0,0)), self.ui.emdbDownloadPushButton.toolTip())
    
    @qtSlot()
    def on_openURLToolButton_clicked(self):
        '''Called when the user clicks the link button
        '''
        
        num = self.ui.emdbNumberLineEdit.text()
        if num == "":
            QtGui.QMessageBox.warning(self, "Warning", "Empty Accession Number")
            return
        QtGui.QDesktopServices.openUrl("http://www.ebi.ac.uk/pdbe/entry/EMD-%s"%num) 
        
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
        if not os.path.exists('data/rawmap/'):
            try:os.makedirs('data/rawmap/')
            except: pass
        self.task=BackgroundTask.launch(self, download_gunzip_task, url, 'data/rawmap/')
    
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
            self.openReference(os.path.abspath(local))
        self.task=None
    
    @qtSlot(QtCore.QModelIndex)
    def on_emdbCannedListView_doubleClicked(self, index):
        ''' Called when the user clicks on the list
        '''
        
        num = index.data(QtCore.Qt.UserRole) #.toString()
        self.ui.emdbNumberLineEdit.setText(num)
        
        text = index.data(QtCore.Qt.UserRole+1)
        self.ui.mapInfoPlainTextEdit.document().setPlainText(text)
    
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
        
        if filename != "" and not os.path.exists(filename):
            QtGui.QMessageBox.warning(self, "Warning", "File does not exist: %s"%filename)

        if filename != "" and os.path.exists(filename):
            img = ndimage_file.read_image(str(filename))
            
            if img.ndim == 3:
                self.ui.referenceWidthLabel.setText(str(img.shape[0]))
                self.ui.referenceHeightLabel.setText(str(img.shape[1]))
                self.ui.referenceDepthLabel.setText(str(img.shape[2]))
            else:
                QtGui.QMessageBox.warning(self, "Warning", "File is not a volume: %s"%str(img.shape))
            header = ndimage_file.read_header(filename)
            self.ui.referencePixelSizeDoubleSpinBox.setValue(header['apix'])

        return filename
    
    def registerPage(self, wizardPage):
        '''
        '''
        
        wizardPage.registerField(wizardPage.wizard().param("raw_reference_file*"), self.ui.referenceLineEdit)
        wizardPage.registerField(wizardPage.wizard().param("curr_apix*"), self.ui.referencePixelSizeDoubleSpinBox, "value", QtCore.SIGNAL('valueChanged(double)'))
                

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