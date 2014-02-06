''' Propery tree/table editor

.. currentmodule:: arachnid.core.gui.property

.. autosummary::
    :nosignatures:
    :toctree: api_generated/
    :template: api_module.rst
    
    ButtonDelegate
    Property
    PropertyDelegate
    PropertyModel
'''


def setView(widget): #ui, or return model and delegate
    ''' Set model and delegate for the given view
    
    :Parameters:

    widget : QWidget
             Tree, List or Table widget to use with the property model/delegate
    '''
    
    import PropertyModel, PropertyDelegate
    
    widget.setModel(PropertyModel.PropertyModel(widget))
    widget.setItemDelegate(PropertyDelegate.PropertyDelegate(widget))