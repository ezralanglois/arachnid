''' Read a set of alignment parameters and organize images

.. Created on Feb 28, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import format, spider_utility, relion_utility
import spider_params
from ..orient import orient_utility
from ..image import ndimage_file
import numpy
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def is_relion_star(filename):
    ''' Test if input filename is in the Relion star format
    
    :Parameters:
    
    filename : str
               Input filename
    
    :Returns:
    
    flag : bool
           True if the format matches the Relion star format
    '''
    
    return format.get_format(filename) == format.star

def read_alignment(filename, image_file, use_3d=False, align_cols=7, force_list=False, ctf_params=False, **extra):
    ''' Read an alignment file and organize images
    
    Supports both SPIDER and Relion alignment files
    
    :Parameters:
    
    filename : str
               Filename for alignment file
    image_file : str
                 Filename for image template - not required for relion
    use_3d : bool
             If True, then translations are RT, otherwise they are TR
    align_cols : int
                 Number of columns in alignment array (must be greater than 7)
    force_list : bool
                 Return filename index tuple list rather than converting
                 to SPIDER ID label array
    ctf_params : bool
                    Read and return CTF params
    extra : dict
            Unused keyword arguments
    
    :Returns:
    
    files : list or tuple
            List of filename/id tuples or a tuple (filename, label array)
    out : array
          2d array where rows are images and columns are alignment
          parameters: psi,theta,phi,inplane,tx,ty,defocus
    param : dict, optional
            CTF params
    '''
    
    if align_cols < 7: raise ValueError, "align_cols must be greater than 7"
    if not isinstance(image_file, str) and len(image_file) == 1: image_file=image_file[0]
    id_len = extra['id_len'] if 'id_len' in extra else 0
    if 'format' in extra: del extra['format']
    if 'numeric' not in extra: extra['numeric']=True
    supports_spider_id="" if not force_list else None
    if is_relion_star(filename):
        align = format.read(filename, **extra)
        param = numpy.zeros((len(align), align_cols))
        files = []
        if use_3d:
            _logger.info("Standard Relion alignment file - leave 3D")
            for i in xrange(len(align)):
                files.append(relion_utility.relion_file(align[i].rlnImageName))
                if supports_spider_id is not None and not spider_utility.is_spider_filename(files[-1][0]):
                    if supports_spider_id == "":
                        supports_spider_id = spider_utility.spider_filepath(files[-1][0])
                    elif supports_spider_id != spider_utility.spider_filepath(files[-1][0]):
                        supports_spider_id=None
                param[i, 0] = align[i].rlnAnglePsi
                param[i, 1] = align[i].rlnAngleTilt
                param[i, 2] = align[i].rlnAngleRot
                #param[i, 3] = align[i].
                param[i, 4] = align[i].rlnOriginX
                param[i, 5] = align[i].rlnOriginY
                param[i, 6] = align[i].rlnDefocusU
        else:
            _logger.info("Standard Relion alignment file - convert to 2D")
            for i in xrange(len(align)):
                files.append(relion_utility.relion_file(align[i].rlnImageName))
                if supports_spider_id is not None and not spider_utility.is_spider_filename(files[-1][0]): 
                    if supports_spider_id == "":
                        supports_spider_id = spider_utility.spider_filepath(files[-1][0])
                    elif supports_spider_id != spider_utility.spider_filepath(files[-1][0]):
                        supports_spider_id=None
                param[i, 1] = align[i].rlnAngleTilt
                param[i, 2] = align[i].rlnAngleRot
                rot, tx, ty = orient_utility.align_param_3D_to_2D_simple(align[i].rlnAnglePsi, align[i].rlnOriginX, align[i].rlnOriginY)
                param[i, 3] = rot
                param[i, 4] = tx
                param[i, 5] = ty
                param[i, 6] = align[i].rlnDefocusU
        if supports_spider_id is not None:
            label = numpy.zeros((len(param), 2), dtype=numpy.int)
            for i in xrange(len(files)): label[i, :] = (spider_utility.spider_id(files[i][0]), files[i][1])
            if label[:, 1].min() > 0: 
                label[:, 1]-=1
            files = (files[0][0], label)
            if label[:, 1].min() < 0: raise ValueError, "Cannot have a negative index"
        if ctf_params:
            ctf_param=dict(cs=align[0].rlnSphericalAberration,
                           voltage=align[0].rlnVoltage,
                           ampcont=align[0].rlnAmplitudeContrast) if len(align) > 0 else {}
    else:
        if ctf_params: ctf_param=spider_params.read(extra['param_file'])
        if 'ndarray' not in extra or not extra['ndarray']: extra['ndarray']=True
        align = read_spider_alignment(filename, **extra)[0]
        param = numpy.zeros((len(align), align_cols))
        if align.shape[1] > 17: 
            param[:, 6] = align[:, 17]
        if use_3d:
            if numpy.sum(align[:, 5]) != 0.0:
                _logger.info("Standard SPIDER alignment file - convert to 3D")
                if numpy.all(align[:,5]==0): 
                    _logger.info("Detected non-standard SPIDER alignment file with no 2D parameters")
                    param[:, :3] = align[:, :3]
                else:
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
            param[:, 1:3] = align[:, 1:3]
            if param.shape[1] > 7:
                param[:, 3:6] = align[:, 5:8]
                
            else:
                _logger.info("Detected non-standard SPIDER alignment file with only angles")
            if numpy.any(align[:,0]!=0): 
                param[:, 3]=-align[:,0]
        if force_list or not spider_utility.is_spider_filename(image_file):
            if align.shape[1] <= 15:
                files = []
                if isinstance(image_file, str) and align[:, 4].max() > ndimage_file.count_images(image_file):
                    for i in xrange(len(align)):
                        files.append( (image_file, int(i+1)) )
                else:
                    idx = align[:, 4].astype(numpy.int)
                    for i in xrange(len(align)):
                        files.append( (image_file, idx[i]) )
            else:
                files=[]
                label = align[:, 15:17].astype(numpy.int)
                for mic, pid in label:
                    files.append((spider_utility.spider_filename(image_file, int(mic), id_len), int(pid)))
        else:
            label = numpy.zeros((len(param), 2), dtype=numpy.int)
            if align.shape[1] <= 15:
                label[:, 0] = spider_utility.spider_id(image_file)
                if isinstance(image_file, str) and align[:, 4].max() > ndimage_file.count_images(image_file):
                    label[:, 1] = numpy.arange(1, len(align)+1)
                else:
                    label[:, 1] = align[:, 4]
            else:
                label[:, :] = align[:, 15:17].astype(numpy.int)
            if label[:, 1].min() > 0: label[:, 1]-=1
            files = (image_file, label)
            if label[:, 1].min() < 0: raise ValueError, "Cannot have a negative index"
    if ctf_params: return files, param, ctf_param
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


