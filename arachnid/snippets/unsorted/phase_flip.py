''' Phase flip a stack

Download to edit and run: :download:`phase_flip.py <../../arachnid/snippets/phase_flip.py>`

.. seealso::

    List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`

To run:

.. sourcecode:: sh
    
    $ python phase_flip.py

.. literalinclude:: ../../arachnid/snippets/phase_flip.py
   :language: python
   :lines: 20-
   :linenos:
'''
from arachnid.core.metadata import spider_params, format
from arachnid.core.parallel import mpi_utility
from arachnid.core.spider import spider,spider_file
#import numpy

if __name__ == '__main__':
    
    #input_stack = ""
    output_stack = ""
    defocus_file = ""
    params_file="params"
    output_select=""
    
    spi = spider.open_session([output_stack], spider_path="", thread_count=0, enable_results=False) # Create a SPIDER session using the extension of the input_volume
    params = spider_params.read(params_file)
    defocus = format.read(defocus_file, numeric=True)
    
    ftimage = None
    ctfimage = None
    ctf = None
    temp_spider_file = mpi_utility.safe_tempfile("temp_spider_file")
    outid=1
    for particle in defocus:
        id, filename = particle.rlnImageName.split('@')
        id = int(id)
        if hasattr(particle, 'rlnDefocusV'):
            defocus = (particle.rlnDefocusU + particle.rlnDefocusV) / 2.0
        else: defocus = particle.rlnDefocusU
        ctf = spi.tf_ct(defocus=defocus, outputfile=ctf, **params)
        #temp_spider_file
        spider_file.copy_to_spider(filename, spi.replace_ext(temp_spider_file), id-1)
        ftimage = spi.ft(temp_spider_file, outputfile=ftimage)           # Fourier transform reference volume
        ctfimage = spi.mu(ftimage, ctf, outputfile=ctfimage)           # Multiply volume by the CTF
        spi.ft(ctfimage, outputfile=(output_stack, outid))
        outid+=1
    #format.write(output_select, numpy.hstack((numpy.arange(1, len(defocus)+1)[:, numpy.newaxis], numpy.ones(len(defocus), 1))))


