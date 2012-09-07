''' Extend the pyqtProperty to store more information for the Property editor

This class adds additional attributes to the pyqtProperty class.


.. currentmodule:: arachnid.core.gui.property.pyqtProperty

.. autoclass:: PyqtProperty
    :members:

.. Created on Dec 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from PyQt4 import QtCore

class PyqtProperty(QtCore.pyqtProperty):
    ''' Extend a pyqtProperty to hold additional information
        
        :Parameters:
        
        type : object
               Type name string or type
        fget : functor
               Getter method
        fset : functor
               Setter method
        freset : functor
               Reset method
        fdel : functor
                Delete method
        doc : str
              Document string (ignored)
        designable : bool
                     Flag property as designable
        scriptable : bool
                     Flag property as scriptable
        stored : bool
                 Flag property as stored
        user : bool
                 Flag property as User created
        constant : bool
                   Flag property as constant
        final : bool
                Flag property as final
        editorHints : dict
                      Editor hints property dictionary
    '''
    
    def __init__(self, index, type, fget=None, fset=None, freset=None, fdel=None, doc=None, designable=True, scriptable=True, stored=True, user=False, constant=False, final=False, editorHints={}):
        ''' Create a PyqtProperty object
        '''
        
        if isinstance(type, str): type = QtCore.QString(type)
        QtCore.pyqtProperty.__init__(self, type, fget, fset, freset, fdel, doc, designable, scriptable, stored, user, constant, final)
        self.editorHints = editorHints
        self.doc = doc
        self.order_index = index
    


    


