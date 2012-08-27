''' Write images in a variety of formats

The image writing interface selects the image format based on the file extension.  The following table
lists each supported extension with its corresponding file format.

+--------------+-------------------------------------+-+--------------+------------------------------+
|  Extension   |   Format                            | | Extension    |   Format                     |
+==============+=====================================+=+==============+==============================+
| rec,  mrc,   |                                     | | img, IMG     |                              |
| MRC,  ALI,   |   `MRC`_                            | | hed, HED     | `IMAGIC`_                    |
| tnf,  TNF,   |                                     | | imagic       |                              |
| ccp4, map    |                                     | | IMAGIC       |                              |
+--------------+-------------------------------------+-+--------------+------------------------------+
| spi, SPI,    |                                     | | spidersingle |                              |
| spider,      |  `Spider Stack`_                    | | SPIDERSINGLE |  `Spider Image`_             |
| SPIDER       |                                     | | singlespider |                              |
|              |                                     | | SINGLESPIDER |                              |
+--------------+-------------------------------------+-+--------------+------------------------------+
| jpg, JPG,    | `JPEG Image Format`_                | |  png,  PNG   | `Portable Network Graphics`_ |
| jpeg, JPEG   |                                     | |              |                              |
+--------------+-------------------------------------+-+--------------+------------------------------+
| vtk, VTK     | `Visualization Toolkit`_            | | xplor, XPLOR | `X-Plor File Format`_        |
+--------------+-------------------------------------+-+--------------+------------------------------+
| h5,   H5,    |`Hierarchical Data Format`_          | | map, MAP     | `ICOS Image Format`_         |
| hdf,  HDF    |                                     | | icos, ICOS   |                              |
+--------------+-------------------------------------+-+--------------+------------------------------+
| am, AM,      | `AMIRA Image Format`_               | | pif,  PIF    | `Portable Image Format`_     |
| amira, AMIRA |                                     | |              | For EM data                  |
+--------------+-------------------------------------+-+--------------+------------------------------+
| lst,  LST    | List of Image Files                 | | pgm,  PGM    | `Portable Gray Map`_         |
| lsx,  LSX    |                                     | |              |                              |
+--------------+-------------------------------------+-+--------------+------------------------------+

Currently, a mixture of EMAN2 and some Appion code is used to write images. Images can either be written
individually or grouped into stacks. Both Spider and HDF support reading and writing multiple images
from and to stacks.

To support `Frealign`_, an MRC volume can be written out by using the `*.vmrc` file extension.

When writing MRC files using EMAN, the image is first flipped.

.. note::
    
    If the third-party libraries are not installed or detected, then the particular write function
    does not return an error, it is merely disabled.

    
.. _`X-Plor File Format`: http://nmr.cit.nih.gov/xplor-nih/
.. _`MRC`: http://www2.mrc-lmb.cam.ac.uk/
.. _`IMAGIC`: http://imagescience.de/imagic/
.. _`Gatan Image Format`: http://www.gatan.com/
.. _`Spider Stack`: http://www.wadsworth.org/spider_doc/spider/docs/spider.html
.. _`Spider Image`: http://www.wadsworth.org/spider_doc/spider/docs/spider.html
.. _`Visualization Toolkit`: http://www.vtk.org/
.. _`Hierarchical Data Format`: http://www.hdfgroup.org/HDF5/
.. _`Portable Gray Map`: http://netpbm.sourceforge.net/doc/pgm.html
.. _`AMIRA Image Format`: http://www.amira.com/
.. _`JPEG Image Format`: http://www.jpeg.org/
.. _`Flexible Image Transport System`: http://heasarc.gsfc.nasa.gov/docs/heasarc/fits.html
.. _`Portable Image Format`: http://cryoem.ucsd.edu/programs.shtm
.. _`Portable Network Graphics`: http://www.libpng.org/pub/png/
.. _`Tagged Image File Format`: http://www.libtiff.org/
.. _`Video For Linux`: http://linuxtv.org/wiki/index.php/Main_Page
.. _`ICOS Image Format`: http://www.mosaic.ethz.ch/research/projects/closed
.. _`Frealign`: http://emlab.rose2.brandeis.edu/frealign

.. Created on Sep 28, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import logging, os, struct

_logger = logging.getLogger(__name__)#, logging.DEBUG)
_logger.setLevel(logging.INFO)

def write_image(filename, img, index=None, disable_auto_delete=False):
    '''Write an image using the given filename
    
    A simple interface to write an image to a file, which may contain a stack of images. If
    the library, such as EMAN2, cannot determine the appropriate type, then the spider format
    will be automatically chosen.
    
    .. sourcecode:: py
    
        >>> from core.image.reader import *
        >>> img6 = read_image("image_stack.spi", 6)
        >>> img0 = read_image("image_stack.spi")
        >>> write_image("output_stack.spi", img6, 0)
        >>> write_image("output_stack.spi", img0, 1)
        
        # Write in Frealign compatible volume MRC format
        >>> write_image("output_volume.vmrc", img0, 0)
        >>> write_image("output_volume.vmrc", img6, 1)
    
    :Parameters:

    filename : string
               The name and path of the file
    img : image-like object
          The image in library native format to be written out.
    index : integer
            Index of the image in a stack
    '''
    
    ext = os.path.splitext(filename)[1]
    if ext == '.vmrc':
        try:
            return write_image_using_scripps_mrc(filename, img, index, disable_auto_delete)
        except:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.exception("Failed to write file using Scripps MRC")
            else:
                _logger.error("Failed to write file using Scripps MRC")
        return
    try:
        return write_image_using_eman2(filename, img, index, disable_auto_delete)
    except:
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.exception("Failed to write file using EMAN2")
        else:
            _logger.error("Failed to write file using EMAN2")
        raise    

def write_stack(filename, imgs, index=0):
    '''Write a stack of images using the given filename, list of images and list of offsets
    
    A simple interface to write a list of images to a file as a particle stack. If the
    library, such as EMAN2, cannot determine the appropriate type, then the spider format
    will be automatically chosen.
    
    .. sourcecode:: py
    
        >>> from core.image.reader import *
        >>> imgs = read_images("image_stack.spi")
        >>> len(imgs)
        50
        >>> write_stack("output_stack.spi", imgs[:10])
        >>> write_stack("output_stack.spi", imgs[20:30], 10)
    
    :Parameters:

    filename : string
               The name and path of the file
    imgs : list
          List of images in library native format to be written out.
    index : integer
            Index of the image in a stack
    '''
    
    for img in imgs:
        write_image(filename, img, index)
        index += 1

try:
    import eman2_utility
    
    def write_image_using_eman2(filename, img, index, disable_auto_delete=False):
        '''Write an image using the given filename and index
        
        A simple interface to write an image to a file using EMAN2, which may contain a stack of images.
        
        :Parameters:

        filename : string
                   The name and path of the file
        img : image-like object
              The image in library native format to be written out.
        index : integer
                Index of the image in a stack
        '''
        
        ext = eman2_utility.Util.get_filename_ext(filename)
        try:
            type = eman2_utility.EMUtil.get_image_ext_type(ext)
        except:
            _logger.debug("Cannot get type of "+filename+" with extension "+str(ext))
            raise
        if type == eman2_utility.EMUtil.ImageType.IMAGE_MRC: img.process_inplace("xform.flip",{"axis":"y"})
        if not eman2_utility.is_em(img): img = eman2_utility.numpy2em(img)
        if type == eman2_utility.EMUtil.ImageType.IMAGE_UNKNOWN: 
            type = eman2_utility.EMUtil.ImageType.IMAGE_SPIDER
            _logger.debug("Converting to spider format - "+filename)
        if index is None: 
            _logger.debug("Converting to single spider format")
            if type == eman2_utility.EMUtil.ImageType.IMAGE_SPIDER: type = eman2_utility.EMUtil.ImageType.IMAGE_SINGLE_SPIDER
            index = 0
        else:
            _logger.debug("Converting to stack spider format")
        if index == 0 and os.path.exists(filename) and not disable_auto_delete: os.unlink(filename)
        img.write_image_c(filename, index, type)
        if type == eman2_utility.EMUtil.ImageType.IMAGE_SINGLE_SPIDER:
            f = open(filename, 'r+b')
            f.seek(26*4)
            f.write(struct.pack('f', 0.0))
            f.close()
        
except:
    _logger.warn("Cannot load EMAN2 - see documentation for more details")
    
try:
    import scripps.mrc as mrc
    
    def write_image_using_scripps_mrc(filename, img, index, disable_auto_delete=False):
        '''Write an image using the given filename and index
        
        A simple interface to write an image to a file using EMAN2, which may contain a stack of images.
        
        :Parameters:

        filename : string
                   The name and path of the file
        img : image-like object
              The image in library native format to be written out.
        index : integer
                Index of the image in a stack
        '''
        
        img.process_inplace("xform.flip",{"axis":"y"})
        if index is None: index = 0
        try: img = eman2_utility.em2numpy(img)
        except: pass
        if index == 0 and not disable_auto_delete:
            mrc.write(img, filename)
        else:
            mrc.append(img, filename)
        
except:
    _logger.warn("Cannot load Scripps MRC writer - see documentation for more details")


    
    