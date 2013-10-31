'''
.. Created on Aug 20, 2013
.. codeauthor:: robertlanglois
'''

import sys

from arachnid.core.app import tracing
from arachnid.core.metadata import format
from arachnid.core.metadata import format_utility
from arachnid.core.metadata import spider_utility
from arachnid.core.image import ndimage_file
from arachnid.core.image import reconstruct
from arachnid.core.image import ctf
from arachnid.core.image import ndimage_utility
import numpy
ctf;

def iter_images(data, idx):
    '''
    '''
    
    for i in idx:
        filename, pid = spider_utility.relion_file(data[i].rlnImageName)
        yield ndimage_file.read_image(filename, pid-1)
    raise StopIteration

def process_image(img, data, apix, **extra):
    '''
    '''
    
    img = ndimage_utility.fourier_shift(img, data[3], data[4])
    ctfimg = ctf.phase_flip_transfer_function(img.shape, data[5], data[6], data[7], voltage=data[8], apix=apix)
    img = ctf.correct(img, ctfimg)
    return img

if __name__ == '__main__':
    data_file = sys.argv[1]
    output = sys.argv[2]
    tracing.configure_logging()
    extra=dict(thread_count=32, apix=1.5844)#, psi='rlnAnglePsi', theta='rlnAngleTilt', phi='rlnAngleRot'
    data = format.read(data_file, numeric=True)
    even = numpy.arange(0, len(data), 2, dtype=numpy.int)
    odd = numpy.arange(1, len(data), 2, dtype=numpy.int)
    ndata, header = format_utility.tuple2numpy(data, convert=spider_utility.relion_id)
    order="rlnAnglePsi,rlnAngleTilt,rlnAngleRot,rlnOriginX,rlnOriginY,rlnDefocusU,rlnSphericalAberration,rlnAmplitudeContrast,rlnVoltage".split(',')
    ndata = ndata[:, ([header.index(v) for v in order])]
    img = ndimage_file.read_image(spider_utility.relion_file(data[0].rlnImageName)[0])
    vol,vol_even,vol_odd = reconstruct.reconstruct3_bp3f_mp(img.shape[0], iter_images(data, even), iter_images(data, odd), ndata[even], ndata[odd], process_image=process_image, **extra)
    ndimage_file.write_image(output, vol)
    ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
    ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)



