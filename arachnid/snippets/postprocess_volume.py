''' Post-process a raw volume for refinement

Download to edit and run: :download:`prepare_volume.py <../../arachnid/snippets/prepare_volume.py>`

To run:

.. sourcecode:: sh
    
    $ python prepare_volume.py

.. literalinclude:: ../../arachnid/snippets/prepare_volume.py
   :language: python
   :lines: 16-
   :linenos:
'''

from arachnid.core.image import ndimage_utility, ndimage_file
from arachnid.core.metadata import spider_params
from arachnid.core.spider import spider

if __name__ == '__main__':
    
    input_volume = "raw.spi"
    output_volume = "vol"
    params_file="params"
    resolution = 12.0 # Set Zero to disable
    pixel_size = 1.2 # If there is a params file, do not set this
    hi_resolution=0.0 # Set Zero to disable
    bw_pass = 0.05
    bw_stop = 0.05
    hp_bw_pass=bw_pass
    hp_bw_stop=bw_stop
    
    mask_type = "tight"
    
    # Tight mask parameters
    
    threshold_for_binary_mask=None # None means find threshold automatically
    number_of_times_to_dialate_binary_mask=1
    size_of_the_gaussian_kernel=3
    standard_deviation_of_gaussian=9
    
    # Spherical Mask parameters
    mask_edge_width = 3
    mask_edge_type = 'C'
    pixel_diameter = 200 # If there is a params file, do not set this
    
    # Invoke a SPIDER session using the extension from input_volume
    spi = spider.open_session([input_volume], spider_path="", thread_count=0, enable_results=False) # Create a SPIDER session using the extension of the input_volume
    if params_file != "":
        params = spider_params.read(params_file)
        pixel_size = params['apix']
        pixel_diameter = params['pixel_diameter']
    
    # Read input volume into memory
    involume = spi.cp(input_volume)
    
    # High pass filter the volume as a reference for the next round in refinement
    if hi_resolution > 0.0:
        sp = pixel_size/hi_resolution
        pass_band = sp-hp_bw_pass
        stop_band = sp+hp_bw_stop
        if pass_band > 0.35: pass_band = 0.4
        if stop_band > 0.4: stop_band = 0.45
        involume = spi.fq(involume, spi.BUTER_HP, pass_band=pass_band, stop_band=stop_band)
    
    # Low pass filter the volume as a reference for the next round in refinement
    if resolution > 0.0:
        sp = pixel_size/resolution
        pass_band = sp-bw_pass
        stop_band = sp+bw_stop
        if pass_band > 0.35: pass_band = 0.4
        if stop_band > 0.4:  stop_band = 0.45
        involume = spi.fq(involume, spi.BUTER_LP, pass_band=pass_band, stop_band=stop_band)
    
    spi.cp(involume, outputfile=output_volume)
    
    if mask_type == "tight":
        img = ndimage_file.read_image(output_volume)                                                     # Read volume from a file
        mask, th = ndimage_utility.tight_mask(img, threshold=threshold_for_binary_mask, ndilate=number_of_times_to_dialate_binary_mask, gk_size=size_of_the_gaussian_kernel, gk_sigma=standard_deviation_of_gaussian)    # Generate a adaptive tight mask
        print "Threshold: ", th                                                                         # Print threshold used to create the tight mask
        ndimage_file.write_image(output_volume, img*mask)                                               # Write the masked volume to output_volume
    elif mask_type == "spherical":
        center = spider.image_size(spi, input_volume)[0]/2+1                                                                   # Center of the volume
        radius = pixel_diameter/2+mask_edge_width/2 if mask_edge_type == 'C' else pixel_diameter/2+mask_edge_width                # Define radius of the mask
        spi.ma(involume, radius, (center, center, center), mask_edge_type, 'C', mask_edge_width, outputfile=output_volume) # Create a spherical mask with a soft edge
