''' Shared library optimized code

.. currentmodule:: arachnid.core.orient.core

.. todo:: add module

'''

try: 
    import _healpix
    _healpix;
except: pass
#    import logging
#    _logger = logging.getLogger(__name__)
#    _logger.setLevel(logging.DEBUG)
#    _logger.addHandler(logging.StreamHandler())
#    _logger.exception("Module failed to load")
    

try: 
    import _transformations
    _transformations;
except: pass

try:
    import _rotation_mapping
    _rotation_mapping;
except: pass