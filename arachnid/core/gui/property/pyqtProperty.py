''' Extend the pyqtProperty to store more information for the Property editor

This class adds additional attributes to the pyqtProperty class.


.. currentmodule:: arachnid.core.gui.property.pyqtProperty

.. autoclass:: PyqtProperty
    :members:

.. Created on Dec 2, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from ..util.qt4_loader import qtProperty

try:
    class PyqtProperty(qtProperty):
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
            
            qtProperty.__init__(self, type, fget, fset, freset, fdel, doc, designable, scriptable, stored, user, constant, final)
            self.editorHints = editorHints
            self.doc = doc
            self.order_index = index
except:
    #raise # TODO: Need to find a hack for PySide
    def PyqtProperty(index, type, fget=None, fset=None, freset=None, fdel=None, doc=None, notify=None, designable=True, scriptable=True, stored=True, user=False, constant=False, final=False, editorHints={}):
        '''
        '''
        
        prop=qtProperty(type, fget, fset, freset, fdel, doc, notify, designable, scriptable, stored, user, constant, final)
        prop.__doc__=doc
        #setattr(prop, 'editorHints', editorHints)
        #prop.editorHints=editorHints
        return prop
    
    
    
    


    


