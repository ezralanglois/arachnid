''' Convert a Freealign parameter file to SPIDER

Download to edit and run: :download:`frealign2spi.py <../../arachnid/snippets/frealign2spi.py>`

.. seealso::

    List of :py:class:`SPIDER Commands <arachnid.core.spider.spider.Session>`

To run:

.. sourcecode:: sh
    
    $ python frealign2spi.py

.. literalinclude:: ../../arachnid/snippets/frealign2spi.py
   :language: python
   :lines: 21-
   :linenos:
'''
import sys
sys;
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

from arachnid.core.metadata import format, format_utility, spider_utility
import glob

if 1 == 0:
    import logging
    format._logger.setLevel(logging.DEBUG)
    format.frealign._logger.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    format._logger.addHandler(logging.StreamHandler())
    format.frealign._logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':

    # Parameters
    
    frealgin_file = ""
    defocus_file = ""
    map_file = ""
    output_file = ""
    
    defocus = format.read(defocus_file, numeric=True)
    idmap = {}
    for v in format.read(map_file, numeric=True):
        idmap[v.araSpiderID] = spider_utility.spider_id(v.araLeginonFilename)
    

    frealgin = []
    for filename in glob.glob(frealgin_file):
        frealgin.extend(format.read(filename, numeric=True))
    
    newdefocus=[]
    frealgin=format_utility.map_object_list(frealgin, 'film')
    for i in xrange(len(defocus)):
        id = idmap[defocus[i].id]
        try:
            f = frealgin[id]
        except: continue
        newdefocus.append(defocus[i]._replace(defocus=(f.defocusu+f.defocusv)/2))
    
    format.write(output_file, newdefocus)
