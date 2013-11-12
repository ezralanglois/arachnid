''' Generate a single aligned stack from a relion selection file

.. Created on Apr 22, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from ..core.app import program
from ..core.metadata import format, spider_utility, spider_params
from ..core.image import ndimage_file, rotate, ctf
from ..core.orient import orient_utility, healpix
import logging, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def batch(files, output, bin_factor=1.0, resolution=2, align_only=False, use_rtsq=False, use_ctf=False, **extra):
    ''' Reconstruct a 3D volume from a projection stack (or set of stacks)
    
    :Parameters:
    
    files : list
            List of input filenames
    output : str
             Output filename for alignment file
    extra : dict
            Unused keyword arguments
    '''
    
    relion_data = format.read(files[0], numeric=True)
    _logger.info("Processing: %s with %d projections with bin factor %f"%(files[0], len(relion_data), bin_factor))
    values = numpy.zeros((len(relion_data), 18))
    img = None
    has_subset = hasattr(relion_data[0], 'rlnRandomSubset')
    has_class = hasattr(relion_data[0], 'rlnClassNumber')
    has_ang = hasattr(relion_data[0], 'rlnAngleTilt')
    
    angle_map={}
    ref_index=0
    for index, projection in enumerate(relion_data):
        stack_file, stack_id = spider_utility.relion_file(projection.rlnImageName)
        if not align_only:
            img = ndimage_file.read_image(stack_file, stack_id-1)
        if has_ang:
            if resolution>0:
                pix = healpix.ang2pix(resolution, numpy.deg2rad(projection.rlnAngleTilt), numpy.deg2rad(projection.rlnAngleRot))
                theta, phi = numpy.rad2deg(healpix.pix2ang(resolution, pix))
                rang = rotate.rotate_euler((0, theta, phi), (projection.rlnAnglePsi, projection.rlnAngleTilt, projection.rlnAngleRot))
                psi = (rang[0]+rang[2])
                psi, dx, dy = orient_utility.align_param_3D_to_2D_simple(psi, projection.rlnOriginX/bin_factor, projection.rlnOriginY/bin_factor)
                if index < 10:
                    _logger.info("%f,%f,%f -> %f,%f,%f | %f,%f == %f,%f"%(projection.rlnAnglePsi, projection.rlnOriginX/bin_factor, projection.rlnOriginY/bin_factor, psi, dx, dy, projection.rlnAngleTilt, projection.rlnAngleRot, theta, phi))
                
                if img is not None:
                    img = rotate.rotate_image(img, psi, dx, dy)
            else:
                psi, dx, dy = orient_utility.align_param_3D_to_2D_simple(projection.rlnAnglePsi, projection.rlnOriginX/bin_factor, projection.rlnOriginY/bin_factor)
                theta = projection.rlnAngleTilt
                phi = projection.rlnAngleRot
                pix=0
                if theta not in angle_map:
                    angle_map[theta]={}
                if has_class:
                    if phi not in angle_map[theta]:
                        angle_map[theta][phi]={}
                    cl = projection.rlnClassNumber
                    if cl not in angle_map[theta][phi]:
                        ref_index += 1
                        angle_map[theta][phi][cl] = ref_index
                else:
                    if phi not in angle_map[theta]:
                        angle_map[theta][phi]={}
                        ref_index += 1
                        angle_map[theta][phi] = ref_index
                if has_class:
                    pix=angle_map[theta][phi][cl]
                else:
                    pix=angle_map[theta][phi]
                
                if use_ctf:
                    ctfimg = ctf.phase_flip_transfer_function(img.shape, projection.rlnDefocusU, **extra)
                    img = ctf.correct(img, ctfimg)
                if use_rtsq:
                    if img is not None:
                        img = rotate.rotate_image(img, psi, dx, dy)
            if img is not None:
                ndimage_file.write_image(output, img, index)
        else: theta, phi, pix, psi, dx, dy = 0, 0, 0, 0, 0, 0
        values[index, 1] = theta
        values[index, 2] = phi
        values[index, 3] = pix
        values[index, 4] = index+1
        values[index, 5] = psi
        values[index, 6] = dx
        values[index, 7] = dy
        if has_subset: values[index, 14] = projection.rlnRandomSubset
        values[index, 15] = spider_utility.spider_id(stack_file)
        values[index, 16] = stack_id
        values[index, 17] = projection.rlnDefocusU
    
    _logger.info("Writing alignment file")
    format.write(output, values, header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror,micrograph,stack_id,defocus".split(','), prefix="align_")
    if has_subset:
        format.write(output, numpy.vstack((numpy.argwhere(values[:, 14]==1).ravel()+1, numpy.ones(numpy.sum(values[:, 14]==1)))).T, header="id,select".split(','), prefix="h1_sel_")
        format.write(output, numpy.vstack((numpy.argwhere(values[:, 14]==2).ravel()+1, numpy.ones(numpy.sum(values[:, 14]==2)))).T, header="id,select".split(','), prefix="h2_sel_")
        #format.write(output, values[values[:, 14]==1], header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror".split(','), prefix="h1_align_")
        #format.write(output, values[values[:, 14]==2], header="epsi,theta,phi,ref_num,id,psi,tx,ty,nproj,ang_diff,cc_rot,spsi,sx,sy,mirror".split(','), prefix="h2_align_")
    _logger.info("Completed")

def setup_options(parser, pgroup=None, main_option=False):
    #Setup options for automatic option parsing
    
    if pgroup is None: pgroup=parser
    if main_option:
        pgroup.add_option("-i", input_files=[], help="List of input images or stacks named according to the SPIDER format", required_file=True, gui=dict(filetype="file-list"))
        pgroup.add_option("-o", output="",      help="Base filename for output volume and half volumes, which will be named raw_$output, raw1_$output, raw2_$output", gui=dict(filetype="save"), required_file=True)
        pgroup.add_option("-r", resolution=0,   help="Healpix resolution")
        pgroup.add_option("-a", align_only=False,   help="Write alignment file only")
        pgroup.add_option("-d", use_rtsq=False,   help="Rotate/translate output stack")
        pgroup.add_option("",   use_ctf=False,   help="Phase flip the output stack")

#def check_options(options, main_option=False):
    #Check if the option values are valid
    #from ..core.app.settings import OptionValueError
    

def main():
    #Main entry point for this script
    
    program.run_hybrid_program(__name__,
        description = '''Generate an aligned stack from a relion star file
                        
                        $ %prog relion_star_file.star -o dala.ter
                        
                        Note that the output will be two files
                        
                        1. data.ter - aligned and phaseflipped stack
                        2. align_dala.ter - alignment parameters (3 Euler angles)
                        
                        http://
                        
                        Uncomment (but leave a space before) the following lines to run current configuration file on
                        
                        source /guam.raid.cluster.software/arachnid/arachnid.rc
                        nohup %prog -c $PWD/$0 > `basename $0 cfg`log &
                        exit 0
                      ''',
        supports_MPI=False,
        use_version = False,
    )
def dependents(): return [spider_params]
if __name__ == "__main__": main()

