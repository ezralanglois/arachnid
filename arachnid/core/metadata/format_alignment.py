''' Read a set of alignment parameters and organize images

.. Created on Feb 28, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import format, spider_utility, format_utility
import numpy
from ..orient import orient_utility
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def read_alignment(filename, image_file, use_3d=False, **extra):
    ''' Read an alignment file and organize images
    
    Supports both SPIDER and relion
    
    
    '''
    
    id_len = extra['id_len'] if 'id_len' in extra else 0
    if 'format' in extra: del extra['format']
    if 'numeric' not in extra: extra['numeric']=True
    if format.get_format(filename, **extra) == format.star:
        align = format.read(filename, **extra)
        param = numpy.zeros((len(align), 6))
        files = []
        if use_3d:
            _logger.info("Standard Relion alignment file - leave 3D")
            for i in xrange(len(align)):
                files.append(spider_utility.relion_file(align[i].rlnImageName))
                param[i, 0] = align[i].rlnAnglePsi
                param[i, 1] = align[i].rlnAngleTilt
                param[i, 2] = align[i].rlnAngleRot
                #param[i, 3] = align[i].
                param[i, 4] = align[i].rlnOriginX
                param[i, 5] = align[i].rlnOriginY
        else:
            _logger.info("Standard Relion alignment file - convert to 2D")
            for i in xrange(len(align)):
                files.append(spider_utility.relion_file(align[i].rlnImageName))
                param[i, 1] = align[i].rlnAngleTilt
                param[i, 2] = align[i].rlnAngleRot
                rot, tx, ty = orient_utility.align_param_3D_to_2D_simple(align[i].rlnAnglePsi, align[i].rlnOriginX, align[i].rlnOriginY)
                param[i, 3] = rot
                param[i, 4] = tx
                param[i, 5] = ty
            
    else:
        align = format_utility.tuple2numpy(read_spider_alignment(filename, **extra))[0]
        param = numpy.zeros((len(align), 6))
        if use_3d:
            if numpy.sum(align[:, 5]) != 0.0:
                _logger.info("Standard SPIDER alignment file - convert to 3D")
                for i in xrange(len(align)):
                    rot, tx, ty = orient_utility.align_param_2D_to_3D_simple(align[i, 5], align[i, 6], align[i, 7])
                    param[i, 0] = rot
                    param[i, 1:3] = align[i, 1:3]
                    param[i, 4:6] = (tx, ty)
            else:
                _logger.info("Detected non-standard SPIDER alignment file with only angles")
                param[:, :3] = align[:, :3]
        else:
            _logger.info("Standard SPIDER alignment file - leave 2D")
            param[:, :3] = align[:, :3]
            param[:, 3:] = align[:, 5:8]
        if align.shape[1] == 15:
            files = []
            for i in xrange(len(align)):
                files.append( (image_file, int(i+1)) )
        else:
            files=[]
            label = align[:, 15:17].astype(numpy.int)
            for mic, pid in label:
                files.append((spider_utility.spider_filename(image_file, int(mic), id_len), int(pid)))
    return files, param

def read_spider_alignment(filename, header=None, **extra):
    ''' Read a SPIDER alignment data from a file
    
    :Parameters:
    
    filename : str
              Input filename containing alignment data
    header : str
             User-specified header for the alignment file
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    align : list
            List of named tuples
    '''
    
    
    align_header = [
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus",
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id",
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror"
                ]
    if 'numeric' in extra: del extra['numeric']
    align = None
    for h in align_header:
        try:
            align = format.read(filename, numeric=True, header=h, **extra)
        except: pass
        else: break
    if align is None:
        align = format.read(filename, numeric=True, header=header, **extra)
    return align


