''' Run an sanity check on the current installation of Arachnid

This script ensures Arachnid is properly installed and will function correctly.

.. Created on Mar 23, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.spider import spider
import logging
import traceback
import sys
import os

def check_shared_libs():
    '''
    '''
    
    sharedlibs=['arachnid.core.image.spi._spider_ctf', 
                'arachnid.core.image.spi._spider_filter', 
                'arachnid.core.image.spi._spider_interpolate', 
                'arachnid.core.image.spi._spider_reconstruct', 
                'arachnid.core.image.spi._spider_reproject', 
                'arachnid.core.image.spi._spider_rotate', 
                'arachnid.core.image.util._image_utility',
                'arachnid.core.image.util._resample',
                'arachnid.core.image.util._ctf',
                'arachnid.core.learn.core._fastdot', 
                'arachnid.core.orient.core._healpix', 
                'arachnid.core.orient.core._transformations', 
                'arachnid.core.parallel.core._omp']
    
    failed=[]
    for slib in sharedlibs:
        try:
            __import__(slib)
        except: 
            formatted_lines = traceback.format_exc().splitlines()
            if sys.platform == 'darwin':
                failed.append(formatted_lines[-3])
            else:
                failed.append(formatted_lines[-1])
    return failed

def check_spider():
    '''
    '''
    
    warn=[]
    path = spider.determine_spider('/guam.raid.cluster.software/spider.21.00')
    if path == "": path = spider.determine_spider(os.path.dirname(sys.argv[0]))
    if path == "":
        warn.append('Cannot find an installed version of SPIDER - you will be asked to find it later')
    return warn

def check_qt():
    '''
    '''
    
    failed=[]
    try:
        from ..core.gui.util import qtapp
        qtapp.create_app()
    except:
        formatted_lines = traceback.format_exc().splitlines()
        if sys.platform == 'darwin':
            failed.append(formatted_lines[-3])
        else:
            failed.append(formatted_lines[-1])
    return failed

def sanitycheck():
    ''' Sanity check code for testing the installation
    '''
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    failed = []
    failed.extend(check_shared_libs())
    failed.extend(check_qt())
    warn = []
    warn.extend(check_spider())
    
    if len(failed) > 0: 
        for msg in warn:
            logging.warn(msg)
        for msg in failed:
            logging.error(msg)
        logging.error("%d tests failed"%len(failed))
    elif len(warn) > 0: 
        for msg in warn:
            logging.warn(msg)
        logging.warn("%d warnings"%len(warn))
    else:
        logging.info("All tests completed successfully")
    

if __name__ == "__main__": sanitycheck()

