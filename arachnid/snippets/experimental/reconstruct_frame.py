'''
.. Created on Aug 20, 2013
.. codeauthor:: robertlanglois
'''

import sys

from arachnid.core.app import tracing
from arachnid.core.metadata import format
from arachnid.core.metadata import format_utility
from arachnid.core.metadata import spider_utility, relion_utility
from arachnid.core.image import ndimage_file,ndimage_utility
from arachnid.core.image import reconstruct
from arachnid.core.image import ctf
from arachnid.core.image import ndimage_filter
import numpy
ctf;

def iter_images(data, frames, rng, idx):
    '''
    '''
    
    for i in idx:
        fid, pid = relion_utility.relion_file(data[i].rlnImageName)
        filename = spider_utility.spider_filename(frames, fid)
        if hasattr(rng, '__iter__'):
            avg = None
            for rframe in rng:
                img = ndimage_file.read_image(spider_utility.frame_filename(filename, rframe), pid-1)
                if avg is None: avg = img
                else: avg += img
            yield avg
        else:
            if i == idx[0]:
                print data[i].rlnImageName
                print filename
                print spider_utility.frame_filename(filename, rng)
            yield ndimage_file.read_image(spider_utility.frame_filename(filename, rng), pid-1)
    raise StopIteration

def process_image(img, data, apix, mask, norm_mask, **extra):
    '''
    '''
    
    img=ndimage_filter.ramp(img)
    ndimage_utility.replace_outlier(img, 2.5, out=img)
    #img = ndimage_filter.histfit(img, mask, noise_win)
    ndimage_utility.normalize_standard(img, norm_mask, out=img)
    img = ndimage_utility.fourier_shift(img, data[3], data[4])
    ctfimg = ctf.phase_flip_transfer_function(img.shape, data[5], data[6], data[7], voltage=data[8], apix=apix)
    img = ctf.correct(img, ctfimg)
    return img

if __name__ == '__main__':
    data_file = sys.argv[1]
    frames = sys.argv[2]
    output = sys.argv[3]
    rng = sys.argv[4] if len(sys.argv)>4 else 1
    mode = 2
    
    try: rng=int(rng)
    except:
        try:
            rng = tuple([int(v) for v in rng.split('-')])
            print rng
            rng = xrange(rng[0], rng[1])
        except:
            rng = tuple([int(v) for v in rng.split(',')])
            print rng
    tracing.configure_logging()
    extra=dict(thread_count=32, apix=1.5844, radius=100)#, psi='rlnAnglePsi', theta='rlnAngleTilt', phi='rlnAngleRot'
    data = format.read(data_file, numeric=True)
    even = numpy.arange(0, len(data), 2, dtype=numpy.int)
    odd = numpy.arange(1, len(data), 2, dtype=numpy.int)
    idmap={}
    for i in xrange(len(data)):
        filename, id = relion_utility.relion_id(data[i].rlnImageName, 0, False)
        if filename not in idmap: idmap[filename]=0
        idmap[filename] += 1
        if i < 20:
            print data[i].rlnImageName, ' -> ', relion_utility.relion_identifier(filename, idmap[filename])
        data[i] = data[i]._replace(rlnImageName=relion_utility.relion_identifier(filename, idmap[filename]))
    ndata, header = format_utility.tuple2numpy(data, convert=relion_utility.relion_id)
    order="rlnAnglePsi,rlnAngleTilt,rlnAngleRot,rlnOriginX,rlnOriginY,rlnDefocusU,rlnSphericalAberration,rlnAmplitudeContrast,rlnVoltage".split(',')
    ndata = ndata[:, ([header.index(v) for v in order])]
    # Back project
    img = ndimage_file.read_image(frames)
    
    extra['mask']=ndimage_utility.model_disk(extra['radius'], img.shape)
    extra['norm_mask']=extra['mask']*-1+1
    
    if mode == 2: # Single frame reconstruction
        vol,vol_even,vol_odd = reconstruct.reconstruct3_bp3f_mp(img.shape[0], iter_images(data, frames, rng, even), iter_images(data, frames, rng, odd), ndata[even], ndata[odd], process_image=process_image, **extra)
        # Write volume to file
        ndimage_file.write_image(output, vol)
        ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
        ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)
    if mode == 1: # Single frame reconstructions
        rng_list=rng
        for rng in rng_list:
            output = spider_utility.spider_filename(output, rng)
            vol,vol_even,vol_odd = reconstruct.reconstruct3_bp3f_mp(img.shape[0], iter_images(data, frames, rng, even), iter_images(data, frames, rng, odd), ndata[even], ndata[odd], process_image=process_image, **extra)
            # Write volume to file
            ndimage_file.write_image(output, vol)
            ndimage_file.write_image(format_utility.add_prefix(output, "even_"), vol_even)
            ndimage_file.write_image(format_utility.add_prefix(output, "odd_"), vol_odd)


