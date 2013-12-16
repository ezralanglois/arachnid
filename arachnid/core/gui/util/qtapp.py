''' Utilities for Qt Applications

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

try: 
    from qt4_loader import QtGui, QtCore
    QtGui;
except:
    QtGui=None

def create_app():
    ''' Create an instance of application with the appropriate 
    properties.
    
    :Returns:
    
    app : QtGui.QApplication
          Instance of QApplication
    '''
    import arachnid
    if QtGui is None: return None
    #QtGui.QApplication.setStyle('cde')
    app = QtGui.QApplication([])
    QtCore.QCoreApplication.setOrganizationName("Arachnid")
    QtCore.QCoreApplication.setOrganizationDomain(arachnid.__url__)
    QtCore.QCoreApplication.setApplicationName(arachnid.__project__)
    QtCore.QCoreApplication.setApplicationVersion(arachnid.__version__)
    #QtCore.QSettings.setPath(QtCore.QSettings.Format.NativeFormat, QtCore.QSettings.Scope.UserScope, "$HOME/.arachnid")
    return app
