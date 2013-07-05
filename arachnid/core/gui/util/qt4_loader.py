

try: 
    from PySide import QtGui, QtCore
    QtGui, QtCore;
    raise ImportError, "Dum"
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
    QtGui, QtCore;
    qtSignal=QtCore.pyqtSignal
    qtSlot= QtCore.pyqtSlot
    qtProperty= QtCore.pyqtProperty
