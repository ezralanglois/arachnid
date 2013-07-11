import glob,sys,os
from subprocess import call
import subprocess
try:
    files = glob.glob(sys.argv[1])
    gain=sys.argv[2]
    cl = sys.argv[3:]
except:
    print 'Usage: python alignmic_gpu.py "input_stacks*.mrc" gain_image.mrc -fod <stack_size>/4 [other options for dosefgpu_driftcorr]'
    call(['dosefgpu_driftcorr'], shell=True)
    sys.exit(1)

#    Input1 : InputStack.mrc OutputStack.mrc Gain.mrc GPUID
#    Input2 : InputStack.mrc OutputStack.mrc Gain.mrc GPUID Dark.mrc

#-fod to TotalNumFrame/4

def ncall(args):
    print " ".join(args)
    if 1 == 0:
        sp = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = sp.communicate()
        if out:
                print "standard output of subprocess:"
                print out
        if err:
                print "standard error of subprocess:"
                print err
        return sp.returncode
    else:
        return call(args)

for f in files:
    print "Processing", f
    o='gc_'+os.path.basename(f)
    if os.path.exists('align_'+os.path.basename(f)): continue
    if ncall(["dosefgpu_flat", f, o, gain, '0']) != 0:
        print "Gain correction failed", f
        continue
    
        #f=open(o,'rb')
        #h=f.read(224)
        #dim=struct.unpack('=' + 'i'*10 + 'f'*6 + 'i'*3 + 'f'*3 + 'i'*30 + '4s'*2 + 'fi', h)[:3]   #(dimx,dimy,dimz)
        #f.close()
    
    if ncall(["dosefgpu_driftcorr", o, "-fcs", 'align_'+os.path.basename(f)]) != 0:
        raise ValueError, "Alignment failed"
    os.unlink(o)