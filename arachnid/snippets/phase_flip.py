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
from arachnid.core.metadata import spider_params, format, spider_utility
from arachnid.core.spider import spider

if __name__ == '__main__':
    
    #input_stack = ""
    output_stack = ""
    defocus_file = ""
    params_file="params"
    
    spi = spider.open_session([input_stack], spider_path="", thread_count=0, enable_results=False) # Create a SPIDER session using the extension of the input_volume
    params = spider_params.read(params_file)
    defocus = format.read(defocus_file, numeric=True)
    
    ctf = None
    for particle in defocus_file:
        id, filename = particle.rlnImageName.split('@')
        defocus = (particle.rlnDefocusU + rlnDefocusV) / 2.0
        ctf = session.tf_ct(defocus=defocus, outputfile=ctf, **params)
        ftimage = session.ft((filename, id), outputfile=ftimage)           # Fourier transform reference volume
        ctfimage = session.mu(ftimage, ctf, outputfile=ctfimage)           # Multiply volume by the CTF
        session.ft(ctfimage, outputfile=(output_stack, id))


