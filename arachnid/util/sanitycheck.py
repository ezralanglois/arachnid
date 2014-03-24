''' Run an sanity check on the current installation of Arachnid

This script ensures Arachnid is properly installed and will function correctly.

.. Created on Mar 23, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging
import traceback

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
    
    failed=[]
    for slib in sharedlibs:
        try:
            __import__(slib)
        except: failed.append(slib)
    if len(failed) > 0: 
        for slib in failed:
            try: 
                __import__(slib)
            except: 
                formatted_lines = traceback.format_exc().splitlines()
                logging.error(formatted_lines[-3])
        logging.error("Failed to load %d shared libraries"%len(failed))
    else:
        logging.info("All tests completed successfully")
    

if __name__ == "__main__": sanitycheck()

