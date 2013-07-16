''' Low-level binding of SPIDER code to Python

.. currentmodule:: arachnid.core.image.spi


Spider filters (_spider_filter)
===============================

.. function:: ramp(img)

   Remove change in illumination across an image
    
   :param img: Input/output image
   :type img: array
   :return: Tuple of image and exit code (0 means success)
   :rtype: array, int
'''

try: 
    import _spider_reconstruct
    _spider_reconstruct;
except:pass

try: 
    import _spider_reproject
    _spider_reproject;
except:pass

try: 
    import _spider_rotate
    _spider_rotate;
except:pass

try: 
    import _spider_align
    _spider_align;
except:pass


try: 
    import _spider_interpolate
    _spider_interpolate;
except:pass

try: 
    import _spider_rotate_dist
    _spider_rotate_dist;
except:pass

try: 
    import _spider_filter
    _spider_filter;
except:pass




