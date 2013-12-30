'''
.. Created on Dec 26, 2013
.. codeauthor:: robertlanglois
'''
import sys
import EMAN2



if __name__ == '__main__':

    # Parameters
    proj_stack = sys.argv[1]
    output = sys.argv[2]
    
    e = EMAN2.EMData()
    
    fout = open(output, 'w')
    
    for i in xrange(EMAN2.EMUtil.get_image_count(proj_stack)):
        e.read_image(proj_stack, i)
        vals =[]
        if i == 0:
            fout.write(",".join(['id', 'mean']+['c_%d'%k for k in xrange(1, e.get_xsize())]))
            fout.write('\n')
        for j in xrange(e.get_xsize()):
            vals.append("%e"%e.get_value_at(j))
        fout.write("%d,"%(i+1))
        fout.write(",".join(vals)+"\n")
    fout.close()

    