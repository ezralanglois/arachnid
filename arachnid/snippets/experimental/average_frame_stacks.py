''' Average neighboring frames

Download to edit and run: :download:`average_frame_stacks.py <../../arachnid/snippets/average_frame_stacks.py>`

To run:

.. sourcecode:: sh
    
    $ python average_frame_stacks.py relion_sel.star

.. literalinclude:: ../../arachnid/snippets/average_frame_stacks.py
   :language: python
   :lines: 16-
   :linenos:
'''

"""
from arachnid.core.metadata import format, relion_utility, spider_utility, format_utility
from arachnid.core.image import ndimage_file
from collections import defaultdict
import sys

if __name__ == '__main__':

    # Parameters
    selfile = sys.argv[1]
    navg=2
    prefix='avg%d_'%navg
    
    
    vals = format.read(selfile)
    
    idmap={}
    nvals = []
    #spider_utility.frame_filename(filename, rframe)
    k=0
    avg = ndimage_file.read_image(relion_utility.relion_file(vals[0].rlnImageName))
    
    valmap = defaultdict(list)
    for v in vals:
        valmap[relion_utility.relion_id(v.rlnImageName)].append(v)
    
    flag=True
    for v in valmap.itervalues():
        fmap = dict([(spider_utility.frame_id(f.rlnImageName), f) for f in v])
        keys = fmap.keys().sort()
        if flag:
            print keys
            flag=False
        for i in xrange(0, len(keys), 2):
            avg[:]=0
            for j in xrange(navg):
                val = fmap[keys[i+j]]
                img = ndimage_file.read_image(relion_utility.relion_file(val.rlnImageName))
                avg += img
            frame_id = spider_utility.frame_id(filename)/navg+1
            filename = format_utility.add_prefix(spider_utility.frame_filename(val.rlnImageName, frame_id), prefix)
            ndimage_file.write_image(filename, avg, idmap[sid]-1)
            nvals.append(val._replace(rlnImageName=relion_utility.relion_identifier(filename, idmap[sid])))
    format.write(format_utility.add_prefix(selfile, prefix), nvals)
            
    '''
    for i in xrange(len(vals)/navg):
        avg[:]=0
        lastid=0
        filename = relion_utility.relion_file(vals[k].rlnImageName, True)
        sid = spider_utility.spider_id(filename)
        pid = relion_utility.relion_file(vals[k].rlnImageName)[1]
        if sid not in idmap: idmap[sid]=0
        idmap[sid] += 1
        val = vals[k]
        for j in xrange(navg):
            filename = relion_utility.relion_file(vals[k].rlnImageName, True)
            frame_id = spider_utility.frame_id(filename)
            if frame_id!=(lastid+1):
                print frame_id, '==', (lastid+1)
            assert(frame_id==(lastid+1))
            assert(pid == relion_utility.relion_file(vals[k].rlnImageName)[1])
            assert(sid == spider_utility.spider_id(filename))
            img = ndimage_file.read_image(relion_utility.relion_file(vals[k].rlnImageName))
            avg += img
            lastid = frame_id
            k += 1
        frame_id = spider_utility.frame_id(filename)/2+1
        filename = format_utility.add_prefix(spider_utility.frame_filename(filename, frame_id), prefix)
        ndimage_file.write_image(filename, avg, idmap[sid]-1)
        nvals.append(val._replace(rlnImageName=relion_utility.relion_identifier(filename, idmap[sid])))
    '''
    format.write(format_utility.add_prefix(selfile, prefix), nvals)
"""     
