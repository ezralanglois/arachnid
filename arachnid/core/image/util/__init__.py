''' Shared library optimized code from SPIDER and rmeasure

.. currentmodule:: arachnid.core.image.util


Spider Utility (_spider_util)
=============================

.. function:: ramp(img)

   Remove change in illumination across an image
    
   :param img: Input/output image
   :type img: array
   :return: Tuple of image and exit code (0 means success)
   :rtype: array, int
   
.. todo:: setup shared library modules and organize functions

'''

try: 
    import _manifold
    _manifold;
except: pass

try: 
    import _spider_util
    _spider_util;
except: pass

try: 
    import _image_utility
    _image_utility;
except: pass