''' Read a set of alignment parameters and organize images

.. Created on Feb 28, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

import format, spider_utility, format_utility
import numpy
#from ..orient import orient_utility

def read_alignment(filename, image_file, **extra):
    ''' Read an alignment file and organize images
    
    Supports both SPIDER and relion
    
    
    '''
    
    id_len = extra['id_len'] if 'id_len' in extra else 0
    if 'format' in extra: del extra['format']
    if format.get_format(filename, **extra) == format.star:
        align = format.read(filename, **extra)
        param = numpy.zeros((len(align), 6))
        files = []
        for i in xrange(len(align)):
            files.append(spider_utility.relion_file(align[i].rlnImageName))
            param[i, 0] = align[i].rlnAnglePsi
            param[i, 1] = align[i].rlnAngleTilt
            param[i, 2] = align[i].rlnAngleRot
            #param[i, 3] = align[i].
            param[i, 4] = align[i].rlnOriginX
            param[i, 5] = align[i].rlnOriginY
    else:
        align = format_utility.tuple2numpy(read_spider_alignment(filename, **extra))[0]
        param = numpy.zeros((len(align), 6))
        param[:, :3] = align[:, :3]
        param[:, 3:] = align[:, 6:9]
        if align.shape[1] == 15:
            files = image_file
        else:
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
                    "epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,defocus",
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


