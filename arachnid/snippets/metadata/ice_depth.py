''' Determines whether a statical correlation exists between sample type and ice thickness

Download to edit and run: :download:`ice_depth.py <../../arachnid/snippets/ice_depth.py>`

To run:

.. sourcecode:: sh
    
    $ python ice_depth.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/ice_depth.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys
#sys.path.append('~/workspace/arachnida/src')
sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')
from arachnid.core.util.matplotlib_nogui import pylab
from arachnid.core.metadata import format, format_utility, relion_utility
import numpy, os
import logging

#format._logger.setLevel(logging.DEBUG)
#format.csv._logger.setLevel(logging.DEBUG)

#class2only_full.star  class6only_empty.star  ice_thickness.csv  remap.star  time_stamp.csv

if __name__ == '__main__':

    # Parameters
    print sys.argv[0]
    ice_file = 'ice_thickness_hole.csv' #sys.argv[1]
    #ice_file = 'ice_thickness_sq.csv' #sys.argv[1]
    map_file = 'remap.star' #sys.argv[2]
    select1_file = 'class2only_full.star' #sys.argv[3]
    select2_file = 'class6only_empty.star' #sys.argv[4]
    output = 'ice_depth.png' #sys.argv[5]
    dpi=1000
    
    logging.basicConfig(log_level=logging.DEBUG)
    # Read a resolution file
    idmap = format.read(map_file, numeric=True)
    icemap = format.read(ice_file, numeric=True)
    icemap = format_utility.map_object_list(icemap, 'filename')
    spimap = format_utility.map_object_list(idmap, 'araSpiderID')
    
    last=6
    select1 = format.read(select1_file, numeric=True)
    select2 = format.read(select2_file, numeric=True)
    idx = numpy.arange(len(select1), dtype=numpy.int)
    numpy.random.shuffle(idx)
    #idx = idx[:len(select2)]
    icedepth1=[]
    skip1=0
    #for i, s in enumerate(select1):
    for i in idx:
        s=select1[i]
        mic = relion_utility.relion_id(s.rlnImageName)[0]
        try:
            leg = spimap[mic].araLeginonFilename
        except:
            #logging.exception("%d - %s"%(mic, str(spimap.keys())))
            skip1 += 1
        else: #12oct26a_43S-DHX-2_00003gr_00078sq_v01_00003hl_00002en.mrc
            leg = os.path.splitext(os.path.basename(leg))[0]
            id = '_'.join(leg.split('_')[:last])
            icedepth1.append(icemap[id].thickness_mean)
    
    logging.error("Selection1 - Skipped: %d of %d"%(skip1, len(select1)))
    icedepth2=[]
    skip2=0
    for i, s in enumerate(select2):
        mic = relion_utility.relion_id(s.rlnImageName)[0]
        try:
            leg = spimap[mic].araLeginonFilename
        except:
            #logging.exception("%d - %s"%(mic, str(spimap.keys())))
            skip2 += 1
        else: #12oct26a_43S-DHX-2_00003gr_00078sq_v01_00003hl_00002en.mrc
            leg = os.path.splitext(os.path.basename(leg))[0]
            id = '_'.join(leg.split('_')[:last])
            icedepth2.append(icemap[id].thickness_mean)
    logging.error("Selection2 - Skipped: %d of %d"%(skip2, len(select2)))
    
    icedepth1 = numpy.asarray(icedepth1)
    icedepth2 = numpy.asarray(icedepth2)
    #n, bins, patches = P.hist( [x0,x1,x2], 10, weights=[w0, w1, w2], histtype='bar')
    pylab.clf()
    pylab.hist([icedepth1, icedepth2], int(numpy.sqrt(float(min(len(icedepth1), len(icedepth2))))), label=[select1_file, select2_file])
    pylab.savefig(format_utility.new_filename(output, suffix='_hole_ice_hist', ext='.png'), dpi=dpi)
    pylab.clf()
    pylab.hist(icedepth1, int(numpy.sqrt(float(min(len(icedepth1), len(icedepth2))))))
    pylab.savefig(format_utility.new_filename(output, suffix='_hole_ice_hist1', ext='.png'), dpi=dpi)
    pylab.clf()
    pylab.hist(icedepth2, int(numpy.sqrt(float(min(len(icedepth1), len(icedepth2))))))
    pylab.savefig(format_utility.new_filename(output, suffix='_hole_ice_hist2', ext='.png'), dpi=dpi)
    
    if 1 == 0:
        last=4
        holemap = {}
        skip1=0
        select1 = format.read(select1_file, numeric=True)
        for s in select1:
            mic = relion_utility.relion_id(s.rlnImageName)[0]
            try:
                leg = spimap[mic].araLeginonFilename
            except:
                #logging.exception("%d - %s"%(mic, str(spimap.keys())))
                skip1 += 1
            else: #12oct26a_43S-DHX-2_00003gr_00078sq_v01_00003hl_00002en.mrc
                leg = os.path.splitext(os.path.basename(leg))[0]
                id = '_'.join(leg.split('_')[2:last])
                if id not in holemap: holemap[id] = [0, 0]
                holemap[id][0]+=1
        logging.error("Selection1 - Skipped: %d of %d"%(skip1, len(select1)))
        skip2=0
        select2 = format.read(select2_file, numeric=True)
        for s in select2:  
            mic = relion_utility.relion_id(s.rlnImageName)[0]
            try:
                leg = spimap[mic].araLeginonFilename
            except:
                skip2 += 1
            else: #12oct26a_43S-DHX-2_00003gr_00078sq_v01_00003hl_00002en.mrc
                leg = os.path.splitext(os.path.basename(leg))[0]
                id = '_'.join(leg.split('_')[2:last])
                if id not in holemap: holemap[id] = [0, 0]
                holemap[id][1]+=1
        logging.error("Selection2 - Skipped: %d of %d"%(skip2, len(select2)))
        
        count = numpy.asarray(holemap.values())
        frac = count[:, 0].astype(numpy.float)/(count[:, 0]+count[:, 1])
        pylab.clf()
        pylab.hist(frac, int(numpy.sqrt(float(len(frac)))))
        pylab.savefig(format_utility.new_filename(output, suffix='_square_hist', ext='.png'), dpi=dpi)
    
    if 1 == 0:
        gridmap1={}
        skip1=0
        select1 = format.read(select1_file, numeric=True)        
        for s in select1:
            mic = relion_utility.relion_id(s.rlnImageName)[0]
            try:
                leg = os.path.splitext(os.path.basename(leg = spimap[mic]))[0]
            except:
                skip1 += 1
            else: #12oct26a_43S-DHX-2_00003gr_00078sq_v01_00003hl_00002en.mrc
                ids = leg.split('_')[2:6]
                curmap=gridmap1
                for id in ids[:len(ids)-1]:
                    if id not in curmap: curmap[id]={}
                    curmap=curmap[id]
                if ids[len(ids)-1] not in curmap: curmap[ids[len(ids)-1]]=[]
                curmap[ids[len(ids)-1]].append(s)
        logging.info("Selection1 - Skipped: %d"%skip1)
        gridmap2={}
        skip2=0
        select2 = format.read(select2_file, numeric=True)        
        for s in select2:
            mic = relion_utility.relion_id(s.rlnImageName)[0]
            try:
                leg = os.path.splitext(os.path.basename(leg = spimap[mic]))[0]
            except:
                skip2 += 1
            else: #12oct26a_43S-DHX-2_00003gr_00078sq_v01_00003hl_00002en.mrc
                ids = leg.split('_')[2:6]
                curmap=gridmap2
                for id in ids[:len(ids)-1]:
                    if id not in curmap: curmap[id]={}
                    curmap=curmap[id]
                if ids[len(ids)-1] not in curmap: curmap[ids[len(ids)-1]]=[]
                curmap[ids[len(ids)-1]].append(s)
        logging.info("Selection2 - Skipped: %d"%skip2)
    
    
    
    
    if 1 == 0:
        idmap = format.read(map_file, numeric=True)
        spimap = format_utility.map_object_list(idmap, 'araSpiderID')
        select1 = format.read(select1_file, numeric=True)
        for s in select1:
            mic = relion_utility.relion_id(s.rlnImageName)[0]
            try:
                spimap[mic]
            except:
                logging.error("Missing: %d"%mic)
    
    
    