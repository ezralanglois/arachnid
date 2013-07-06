''' Reconstruct a volume

Download to edit and run: :download:`reconstruct.py <../../arachnid/snippets/reconstruct.py>`

To run:

.. sourcecode:: sh
    
    $ python reconstruct.py

.. literalinclude:: ../../arachnid/snippets/reconstruct.py
   :language: python
   :lines: 16-
   :linenos:
'''
import sys

#sys.path.append('/home/robertl/tmp/arachnid-0.0.1/')
from arachnid.core.app import tracing
from arachnid.core.metadata import format, format_utility #, spider_utility
from arachnid.core.image import ndimage_file #, eman2_utility
from arachnid.core.image import reconstruct
from arachnid.core.image import manifold
from arachnid.core.orient import transforms
import numpy,logging

if __name__ == "__main__":
    image_file = sys.argv[1]
    align_file = sys.argv[2]
    output = sys.argv[3]
    use_rtsq = False
    thread_count=16
    nn = 1
    cache_file = "cache_recon"
    
    tracing.configure_logging()
    #Note: this code assuming you are reconstructing a dala stack or a translated stack
    
    # Read an alignment file
    
    align = format.read_alignment(align_file)
    logging.error("Reconstructing %d particles"%len(align))
    align,header = format_utility.tuple2numpy(align)
    if nn > 1:
        align_redundant = numpy.zeros((align.shape[0]*(nn+1), align.shape[1]))
        ang = numpy.zeros((len(align), 4))
        for i in xrange(ang.shape[0]):
            euler1 = numpy.deg2rad(align[i, :3])
            ang[i, :] = transforms.quaternion_from_euler(euler1[0], euler1[1], euler1[2], 'rzyz')
        logging.error("here---1")
        nn = manifold.knn_geodesic_cache(ang, nn, cache_file=cache_file)
        logging.error("here---2")
        index_redundant = nn.row.copy()
        for i in xrange(10):
            logging.error("%d: %d, %d, %f -> %s"%(i, nn.row[i], nn.col[i], nn.data[i], str(align[nn.col[i], :3])))
        for i in xrange(nn.shape[0]):
            align_redundant[i, :] = align[nn.col[i]]
            
            # Fix in plane rotation
            if nn.col[i] == nn.row[i]: continue
            frame = transforms.quaternion_inverse(ang[nn.col[i]])
            tmp = numpy.rad2deg(transforms.euler_from_quaternion(transforms.quaternion_multiply(ang[nn.col[i]], frame), 'rzyz'))
            align_redundant[i, 0] = tmp[0]+tmp[2]
            if i == 1:
                logging.error("Theta: %f == %f"%(tmp[1], ang[nn.col[i], 1]-ang[nn.row[i], 1]))
                logging.error("Inplane = %f -> %f"%(align[nn.row[i], 2]-align[nn.col[i], 2], tmp[0]+tmp[2]))
        del nn
    else:
        align_redundant = align
        index_redundant = numpy.arange(len(align), dtype=numpy.int)
    
    iter_single_images = ndimage_file.iter_images(image_file, index_redundant)
    if thread_count < 2:
        vol = reconstruct.reconstruct_nn4(iter_single_images, align_redundant)
    else:
        image_size = ndimage_file.read_image(image_file).shape[0]
        vol = reconstruct.reconstruct_nn4_mp(iter_single_images, image_size, align_redundant, thread_count=thread_count)
    ndimage_file.write_image(output, vol)
