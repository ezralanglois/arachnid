''' Load QT Python binding either PyQt or PySide

This interface allows the use of either PyQt or PySide when interfacing
with the QT graphical user interface (GUI) library.

.. note::
    
    In order to be compatible between interfaces, only API 2 in PyQ is supported. In
    this API, all objects are Pythonic and QT analogs (e.g. QVariant, QString) are 
    not supported.
    
    For more detail see http://qt-project.org/wiki/Differences_Between_PySide_and_PyQt.

.. Created on Jul 6, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

try: 
    from PySide import QtGui, QtCore, QtWebKit
    QtGui, QtCore, QtWebKit;
    #raise ImportError, "Dum"
    qtSignal=QtCore.Signal
    qtSlot= QtCore.Slot
    qtProperty= QtCore.Property
except ImportError:
    import sip
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QUrl', 2)
    
    from PyQt4 import QtGui, QtCore
    try: 
        from PyQt4 import QtWebKit
        QtWebKit;
    except: pass
    QtGui, QtCore
    qtSignal=QtCore.pyqtSignal
    qtSlot= QtCore.pyqtSlot
    qtProperty= QtCore.pyqtProperty