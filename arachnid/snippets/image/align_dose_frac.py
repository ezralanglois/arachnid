''' Align dose fractionated images

Download to edit and run: :download:`align_dose_frac.py <../../arachnid/snippets/align_dose_frac.py>`

To run:

.. sourcecode:: sh
    
    $ python align_dose_frac.py

.. literalinclude:: ../../arachnid/snippets/align_dose_frac.py
   :language: python
   :lines: 16-
   :linenos:
'''
from arachnid.core.metadata import spider_utility
from arachnid.core.image import ndimage_file, ndimage_utility
import glob, numpy, os

if __name__ == '__main__':

    # Parameters
    
    input_image = "mics_spi/mic_60S2CK2-????.dat"
    output_image = "output/align001/avg_0001.dat"
    particle_size=270
    pixel_size = 1.5844
    pixel_radius = int(particle_size/pixel_size * 0.5)
    output_orig = "output/align001/avg_orig_0001.dat"
    
    output_stack=""#"output/align001/mic_stack_01.spi"
    pow_stack="output/align001/pow_stack_0001.dat"
    window_size=256
    overlap=0.5
    pad=1
    
    input_images = glob.glob(input_image) if 1 == 0 else ["mics_spi/mic_60S2CK2-0013.dat"]
    for input_image in input_images:
        radius = int(pixel_radius* 0.25)
        print input_image
        output_image = spider_utility.spider_filename(output_image, input_image)
        if output_stack != "": output_stack = spider_utility.spider_filename(output_stack, input_image)
        if pow_stack != "": pow_stack = spider_utility.spider_filename(pow_stack, input_image)
        if output_orig != "": output_orig = spider_utility.spider_filename(output_orig, input_image)
        input_image = os.path.splitext(input_image)[0]+"???"+os.path.splitext(input_image)[1]
        print glob.glob(input_image)
        imgs = [ndimage_file.read_image(filename) for filename in glob.glob(input_image)]
        if output_stack != "" or pow_stack != "":
            avg = numpy.empty_like(imgs[0])
            avg[:, :] = 0
            for img in imgs: avg += img
            avg /= len(imgs)
            print numpy.mean(avg), numpy.mean(imgs[0]), len(imgs), imgs[0].shape
            if output_orig != "":
                ndimage_file.write_image(output_orig, avg)
            if output_stack != "":
                ndimage_file.write_image(output_stack, avg, 0)
            if pow_stack != "":
                step = max(1, window_size*overlap)
                rwin = ndimage_utility.rolling_window(avg, (window_size, window_size), (step, step))
                npowerspec = ndimage_utility.powerspec_avg(rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3])), pad)
                mask = ndimage_utility.model_disk(npowerspec.shape[0]*(2.0/pixel_radius), npowerspec.shape)
                sel = mask*-1+1
                npowerspec[mask.astype(numpy.bool)] = numpy.mean(npowerspec[sel.astype(numpy.bool)])
                ndimage_file.write_image(pow_stack, npowerspec, 0)
        avg = imgs[0]
        last = numpy.zeros((len(imgs), 2))
        for i in xrange(3):
            template=avg
            avg = numpy.empty_like(imgs[0])
            avg[:, :] = 0
            for j in xrange(len(imgs)):
                img = imgs[j]
                if numpy.alltrue(img == avg): continue
                dimg = ndimage_utility.dog(img, radius)
                davg = ndimage_utility.dog(template, radius)
                cc_map = ndimage_utility.cross_correlate(dimg, davg)
                y,x = numpy.unravel_index(numpy.argmax(cc_map), cc_map.shape)
                w,h = cc_map.shape
                x = w/2-x
                y = h/2-y
                imgs[j] = ndimage_utility.fourier_shift(img, x, y)
                avg+=imgs[j]
                print i+1, x, y
                last[j, :] = x, y
            radius *= 0.5
            if numpy.sum(last)==0: 
                print "No change - stopping at: %d"%(i+1)
                break
            avg /= len(imgs)
            if output_stack != "":
                ndimage_file.write_image(output_stack, avg, i+1)
            if pow_stack != "":
                step = max(1, window_size*overlap)
                rwin = ndimage_utility.rolling_window(avg, (window_size, window_size), (step, step))
                npowerspec = ndimage_utility.powerspec_avg(rwin.reshape((rwin.shape[0]*rwin.shape[1], rwin.shape[2], rwin.shape[3])), pad)
                npowerspec[mask.astype(numpy.bool)] = numpy.mean(npowerspec[sel.astype(numpy.bool)])
                ndimage_file.write_image(pow_stack, npowerspec, i+1)
        ndimage_file.write_image(output_image, avg)
