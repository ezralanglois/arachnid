''' Run an sanity check on the current installation of Arachnid

This script ensures Arachnid is properly installed and will function correctly.

.. Created on Mar 23, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging

def sanitycheck():
    ''' Sanity check code for testing the installation
    '''
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    sharedlibs=['arachnid.core.image.spi._spider_ctf', 
                'arachnid.core.image.spi._spider_filter', 
                'arachnid.core.image.spi._spider_interpolate', 
                'arachnid.core.image.spi._spider_reconstruct', 
                'arachnid.core.image.spi._spider_reproject', 
                'arachnid.core.image.spi._spider_rotate', 
                'arachnid.core.image.util._image_utility',
                'arachnid.core.image.util._resample',
                'arachnid.core.learn.core._fastdot', 
                'arachnid.core.orient.core._healpix', 
                'arachnid.core.orient.core._transformations', 
                'arachnid.core.parallel.core._omp']

    for slib in sharedlibs:
        try:
            __import__(slib)
        except:
            logging.exception("Cannot import %s"%slib)
        else:
            logging.info("Successful import of %s"%slib)
    

if __name__ == "__main__": sanitycheck()

