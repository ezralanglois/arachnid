'''
This module defines all the commands available in SPIDER as Python functions

SPIDER session example
----------------------

Import and create a SPIDER decision with the given extension

    >>> from vispider.core.spider import Session
    >>> spi = Session(ext='spi')

Decimate a micrograph by a factor of 2 and save to a file called `dec2_mic001.spi`

    >>> spi.dc_s('mic001.spi', 2, 'dec2_mic001.spi')

Note that extensions may be left off

    >>> spi.dc_s('mic001', 2, 'dec2_mic001')

Create an decimated in core file called `dec2`:
    
    >>> dec2 = spi.dc_s('mic001.spi', 2)

Creating a SPIDER Batch file
----------------------------

Setup the option parser to support the `bin_factor` flag

    >>> from vispider.core.spider import open_session, setup_options
    >>> import optparse
    >>> parser = optparse.OptionParser()
    >>> setup_options(parser, optparse.OptionGroup, 'dc_s', spider.open_session)

Parse arguments from the command line `python batch.py --bin-factor 2 input1 output`

    >>> param, files = parser.parse_args()
    >>> files
        ['input1', 'output']
    >>> param 
        {'bin_factor': 2, 'spider_path': "", 'data_ext': "spi", 'thread_count'=0}

Create a spider session with the given parameters

    >>> session = open_session(files, **param)

Run the decimate command with the given parameters
    
    >>> sessions.dc_s(files[0], outputfile=files[1], **param)

.. seealso::
    
    `Spider operations <http://www.wadsworth.org/spider_doc/spider/docs/operations_doc.html>`_

.. todo::

    #. tf_cor only works with spider 18.19 or later
    #. add spider version test
    #. template files? stack files?, e.g. pj_3q
    #. selection range? select all?
    #. pj_3q - angle_list must be int 
    #. ap_sh, pk_3d
    #. find test for orsh
    #. or_sh trans_range must be divisible by trans_step - increase if less than box size - ring radius
    #. auto delete in core files using hook
    #. ma - Check order of parameters
    #. li_d - test if image is 2D or 3D
    #. li_d - figure out how to create properly size incore document file

    #. ms - requires new incore file -- this is a bug
    #. ud n - hangs if incore document  does not exist (Spider Version 18)
    #. tf_cor should have version number - performs TF C in order versions
    #. ud ic segfaults with incore document
    #. incore spider doc files? -- looks like NO
    #. CG PH determine if image or volume (assumes volume)
    #. Bug - 19.11 cp requires seelction file for stacks

.. Created on Mar 20, 2011
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import spider_session
from spider_parameter import spider_image, spider_tuple, spider_doc, spider_stack, spider_select, spider_coord_tuple, is_incore_filename
import collections
import logging, os, numpy

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class Session(spider_session.Session):
    ''' Class defines a set of Spider commands for the current spider Session
    
    Using the SPIDER session
    
    Import and create a SPIDER decision with the given extension
    
        >>> from vispider.core.spider import Session
        >>> spi = Session(ext='spi')
    
    Decimate a micrograph by a factor of 2 and save to a file called `dec2_mic001.spi`
    
        >>> spi.dc_s('mic001.spi', 2, 'dec2_mic001.spi')
    
    Note that extensions may be left off
    
        >>> spi.dc_s('mic001', 2, 'dec2_mic001')
    
    Create an decimated in core file called `dec2`:
        
        >>> dec2 = spi.dc_s('mic001.spi', 2)
    
    '''
    
    def __init__(self, *args, **kwargs):
        #Setup spider based on the version
        
        spider_session.Session.__init__(self, *args, **kwargs)
        
        self.extra_check = _logger.getEffectiveLevel() == logging.DEBUG
        v = self.get_version()
        if v[0] < 18 or (v[0] == 18 and v[1] < 18):
            _logger.warn("This version of SPIDER has alignment problems that may limit your resolution")

    def ac(session, inputfile, outputfile=None, **extra):
        '''Computes the auto-correlation function of a picture by using the Fourier transform 
        relationship. The dimension of the picture need not be a power of two (see 'FT' for any restrictions). 
        Works for 2D and 3D.
        
        >> spi = Session()
        >> spi.ac(inputfile, outputfile)
        
        `Original Spider (AC) <http://www.wadsworth.org/spider_doc/spider/docs/man/ac.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        return spider_session.spider_command_fifo(session, 'ac', inputfile, outputfile, "Applying auto-correlation to an image")
    
    def ac_n(session, inputfile, outputfile=None, **extra):
        '''Computes the normalized auto-correlation function of a picture 
        by using the Fourier transform relationship. The dimension of the 
        picture need not be a power of two (see "FT" for any restrictions). 
        Works for 2D and 3D.
        
        `Original Spider (AC N) <http://www.wadsworth.org/spider_doc/spider/docs/man/acn.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        return spider_session.spider_command_fifo(session, 'ac n', inputfile, outputfile, "Applying the normalized auto-correlation to an image")
    
    def ad(session, inputfile, *otherfiles, **extra):
        '''Squares an image/volume, point-by-point.
        
        `Original Spider (AD) <http://www.wadsworth.org/spider_doc/spider/docs/man/ad.html>`_
        
        .. note:: 
        
            With this command you must use outputfile="name of outputfile", 
            e.g. session.ad(inputfile, anotherfile1, anotherfile2, outputfile=outputfile)
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        otherfiles : str
                     Other filenames listed on the as function parameters
        extra : dict
                Unused key word arguments (outputfile hidden here and used)
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_multi_input(session, 'ad', "Add a set of files", inputfile, *otherfiles, **extra)
    
    def ap_ref(session, inputfile, inputselect, reference, selectref, ring_file="", angle_range=0.0, 
               angle_threshold=1.0, trans_range=16, first_ring=5, ring_last=0, ring_step=1, test_mirror=True, refangles=None, 
               inputangles=None, outputfile=None, **extra):
        '''Compares a set of experimental images with a set of reference images. For each 
        experimental image, it finds the in-plane Euler rotation which aligns the experimental 
        image with the most-similar reference image. Then, if translation search is specifed, 
        it finds the X & Y shifts which align the reference image with the rotated experimental 
        image. Can restrict angular range of projections. Can restrict checking of a mirror image. 
        (See align_overview.html for comparison of 'AP' operations.)
        
        `Original Spider (AP REF) <http://www.wadsworth.org/spider_doc/spider/docs/man/apref.html>`_
        
        .. todo::
            
            Bugs: Does not support incore document files!
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image projection stack
        inputselect : str
                      Experiment projection selection file
        reference : str
                    Filename for the reference projection stack
        selectref : str
                    Filename for the reference projection selection file
        ring_file : str
                    Temporary ring file
        angle_range : float
                      Maximum allowed deviation of the Euler angles
        angle_threshold : float
                          Record differences that exceed this threshold
        trans_range : float
                      Maximum allowed translation
        first_ring : int
                     First polar ring to analyze
        ring_last : int
                    Last polar ring to analyze
        ring_step : int
                    Polar ring step size
        test_mirror : bool
                      If true, test the mirror position of the projection
        refangles : str
                    Document file with euler angles for each reference
        inputangles : str
                      Document file with euler angles for each experimental projection (previous alignment)
        outputfile : str
                     Filename of output image (If none, temporary incore file is used and returned)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        _logger.debug("Performing multi-reference alignment")
        if outputfile is None:  raise ValueError, "Incore documents not supported by AP SH"
        else: session.de(outputfile)
        if not isinstance(ring_file, str) or ring_file == "":
            if isinstance(ring_file, int): ring_file = os.path.join(os.path.dirname(inputfile), "scratch_rings_%d"%ring_file)
            elif ring_file=="": ring_file = os.path.join(os.path.dirname(inputfile), "scratch_rings")
        if 1 == 0:
            test_mirror = '(1)' if test_mirror else '(0)'
        elif 1 == 1:
            test_mirror = 'Y' if test_mirror else 'N'
            
            if supports_internal_rtsq(session) and inputangles is not None:
                test_mirror+=",Y"
            #else:
            #    test_mirror+=",N"
        else:
            if supports_internal_rtsq(session) and inputangles is not None:
                test_mirror = spider_tuple(test_mirror, 1)
            else: test_mirror = spider_tuple(test_mirror)
        inputselect, input_count = spider_session.ensure_stack_select(session, inputfile, inputselect)[:2]
        selectref, ref_count = spider_session.ensure_stack_select(session, reference, selectref)[:2]
        session.invoke('ap ref', spider_stack(reference, ref_count), 
                           spider_select(selectref), spider_tuple(trans_range), #, trans_step), 
                           spider_tuple(first_ring, ring_last, ring_step), #, ray_step), 
                           spider_doc(refangles), spider_image(ring_file), spider_stack(inputfile, input_count), 
                           spider_select(inputselect), spider_doc(inputangles), 
                           spider_tuple(angle_range, angle_threshold),
                           test_mirror,
                           spider_doc(outputfile))
        return outputfile
    
    def ap_sh(session, inputfile, inputselect, reference, selectref, angle_range=0.0, 
              angle_threshold=1.0, trans_range=24, trans_step=1, first_ring=1, ring_last=0, ring_step=1, 
              ray_step=1, test_mirror=True, refangles=None, inputangles=None, outputfile=None, **extra):
        '''Compares a series of experimental images with a series of reference images. For each experimental 
        image, it finds the in-plane Euler rotation angle, and X, Y translational shifts which align the image 
        with the most-similar reference image. Exhaustively checks all requested rotations and shifts. Can 
        restrict angular range of projections. Can restrict checking of mirror image. 
                
        `Original Spider (AP SH) <http://www.wadsworth.org/spider_doc/spider/docs/man/apsh.html>`_
        
        .. todo::
            
            Bugs: Does not support incore document files!
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image projection stack
        inputselect : str
                      Experiment projection selection file
        reference : str
                    Filename for the reference projection stack
        selectref : str
                    Filename for the reference projection selection file
        angle_range : float
                      Maximum allowed deviation of the Euler angles, where 0.0 means no restriction
        angle_threshold : float
                          Record differences that exceed this threshold
        trans_range : float
                      Maximum allowed translation; if this value exceeds the window size, then it will lowered to the maximum possible
        trans_step : float
                     Translation step size
        first_ring : int
                     First polar ring to analyze
        ring_last : int
                    Last polar ring to analyze; if this value is zero, then it is chosen to be the radius of the particle in pixels
        ring_step : int
                    Polar ring step size
        ray_step : int
                    Step for the radial array
        test_mirror : bool
                      If true, test the mirror position of the projection
        refangles : str
                    Document file with euler angles for each reference
        inputangles : str
                      Document file with euler angles for each experimental projection (previous alignment)
        outputfile : str
                     Filename of output image (If none, temporary incore file is used and returned)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        _logger.debug("Performing multi-reference alignment")
        if outputfile is None:  raise ValueError, "Incore documents not supported by AP SH"
        session.de(outputfile)
        inputselect, input_count = spider_session.ensure_stack_select(session, inputfile, inputselect)[:2]
        selectref, ref_count = spider_session.ensure_stack_select(session, reference, selectref)[:2]
        if 1 == 0:
            test_mirror = '(1)' if test_mirror else '(0)'
        elif 1 == 1:
            test_mirror = 'Y' if test_mirror else 'N'
            if supports_internal_rtsq(session) and inputangles is not None:
                test_mirror+=",Y"
            #else:
            #    test_mirror+=",N"
        else:
            if supports_internal_rtsq(session) and inputangles is not None:
                test_mirror = spider_tuple(test_mirror, 1)
            else: test_mirror = spider_tuple(test_mirror)
        session.invoke('ap sh', spider_stack(reference, ref_count), 
                       spider_select(selectref), spider_tuple(trans_range, trans_step), 
                       spider_tuple(first_ring, ring_last, ring_step, ray_step), 
                       spider_doc(refangles), spider_stack(inputfile, input_count), 
                       spider_select(inputselect), spider_doc(inputangles), 
                       spider_tuple(angle_range, angle_threshold),
                       test_mirror,
                       spider_doc(outputfile))
        return outputfile

    def ar(session, inputfile, operation, outputfile=None, **extra):
        '''Performs arithmetic operations point for point on the input image to create an output image.
        
        `Original Spider (AR) <http://www.wadsworth.org/spider_doc/spider/docs/man/ar.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        operation : str
                    Math operation as a string
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        _logger.debug("Perform math operation on an image")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        session.invoke('ar', spider_image(inputfile), spider_image(outputfile), operation)
        return outputfile
    
    def bl(session, image_size, background=2.0, outputfile=None, **extra):
        ''' Creates an image/volume with a specified background
                
        `Original Spider (BL) <http://www.wadsworth.org/spider_doc/spider/docs/man/bl.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        image_size : (float,float)
                     Size of the image
        background : float
                     Background color of the image (Default: 2.0)
        outputfile : str
                 Filename of the output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Int or str describing the output file
        '''
        
        _logger.debug("Creating blank image of size %s in file %s: "%(str(image_size), str(outputfile)))
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        session.invoke('bl', spider_image(outputfile), spider_coord_tuple(image_size), 'N', spider_tuple(background))
        return outputfile
    
    def bp_32f(session, inputfile, angle_file, input_select=None, sym_file="", outputfile=None, **extra):
        '''Calculates two sample reconstructions from randomly selected subsets containing 
        half of the total projections and a a total-3D reconstruction from all the projections 
        using back-projection interpolated in Fourier space. This operation is the same as 'BP 3F' 
        with the addition of the two randomly selected sample reconstructions. See: Comparison 
        of 'BP' operations.
                
        `Original Spider (BP 32F) <http://www.wadsworth.org/spider_doc/spider/docs/man/bp32f.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Input file template, stack template or stack
        angle_file : str
                     File containing euler angles and shifts
        stack_count : int
                     Number of projections in the stack
        input_select : int
                       Input filename or number of images in the stack
        sym_file : str
                   Input filename with symetries
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output full volume
        outputfile1 : str
                      Output half volume
        outputfile2 : str
                      Output half volume
        '''
        
        _logger.debug("Back project")
        outputfile = spider_session.ensure_output_recon3(session, outputfile)
        input_select, max_count = spider_session.ensure_stack_select(session, inputfile, input_select)[:2]
        session.invoke('bp 32f', spider_stack(inputfile, max_count), spider_select(input_select), spider_doc(angle_file), spider_doc(sym_file), spider_image(outputfile[0]), spider_image(outputfile[1]), spider_image(outputfile[2]))
        return outputfile
    
    def bp_cg(session, inputfile, angle_file, cg_radius=0, error_limit=1e-5, chi2_limit=0.0, iter_limit=20, reg_mode=1, lambda_weight=2000.0, input_select=None, sym_file="", outputfile=None, pixel_diameter=None, **extra):
        '''Calculates two sample reconstructions from randomly selected subsets 
        containing half of the total projections and a a total-3D reconstruction 
        from all the projections using conjugate gradients with regularization.
                
        `Original Spider (BP CG) <http://www.wadsworth.org/spider_doc/spider/docs/man/bpcg.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input file template, stack template or stack
        angle_file : str
                     File containing euler angles and shifts
        cg_radius : int
                    Radius of reconstructed object
        error_limit : float
                      Stopping criteria
        chi2_limit :  float
                      Stopping criteria
        iter_limit :  float
                      Maximum number of iterations
        reg_mode : int
                   Regularization mode: (0) No regularization (1) First derivative (2) Second derivative (3) Third derivative
        lambda_weight : float
                        Weight of regularization
        stack_count : int
                     Number of projections in the stack
        input_select : int
                       Input filename or number of images in the stack
        sym_file : str
                   Input filename with symetries
        outputfile : str
                     Filename for incore stack (Default: None)
        pixel_diameter : int
                         Pixel diameter of the particle (params file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output full volume
        '''
        
        _logger.debug("SIRT")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        if cg_radius == 0: cg_radius = int(pixel_diameter/2.0) + 1
        input_select, max_count = spider_session.ensure_stack_select(session, inputfile, input_select)[:2]
        session.invoke('bp cg', spider_stack(inputfile, max_count), spider_select(input_select), spider_tuple(cg_radius), spider_doc(angle_file), 'N', 
                       spider_image(outputfile), spider_tuple(error_limit, chi2_limit), spider_tuple(iter_limit, reg_mode), spider_tuple(lambda_weight)) #, spider_tuple('F')
        return outputfile
    
    def bp_cg_3(session, inputfile, angle_file, cg_radius=0, error_limit=1e-5, chi2_limit=0.0, iter_limit=20, reg_mode=1, 
                lambda_weight=2000.0, input_select=None, sym_file="", outputfile=None, pixel_diameter=None, **extra):
        '''Calculates two sample reconstructions from randomly selected subsets 
        containing half of the total projections and a a total-3D reconstruction 
        from all the projections using conjugate gradients with regularization. This 
        operation is the same as 'BP 3G' with the addition of the two randomly 
        selected sample reconstructions. Only works for square projection images.
                
        `Original Spider (BP CG 3) <http://www.wadsworth.org/spider_doc/spider/docs/man/bpcg3.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input file template, stack template or stack
        angle_file : str
                     File containing euler angles and shifts
        cg_radius : int
                    Radius of reconstructed object
        error_limit : float
                      Stopping criteria
        chi2_limit :  float
                      Stopping criteria
        iter_limit :  float
                      Maximum number of iterations
        reg_mode : int
                   Regularization mode: (0) No regularization (1) First derivative (2) Second derivative (3) Third derivative
        lambda_weight : float
                        Weight of regularization
        stack_count : int
                     Number of projections in the stack
        input_select : int
                       Input filename or number of images in the stack
        sym_file : str
                   Input filename with symetries
        outputfile : str
                     Filename for incore stack (Default: None)
        pixel_diameter : int
                         Pixel diameter of the particle
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        values : tuple
                 Namedtuple of values where fields correspond to header names
        '''
        
        _logger.debug("SIRT")
        outputfile = spider_session.ensure_output_recon3(session, outputfile)
        if cg_radius == 0: cg_radius = int(pixel_diameter/2.0) + 1
        input_select, max_count = spider_session.ensure_stack_select(session, inputfile, input_select)[:2]
        session.invoke('bp cg 3', spider_stack(inputfile, max_count), spider_select(input_select), spider_tuple(cg_radius), 
                       spider_doc(angle_file), 'N', spider_image(outputfile[0]), spider_image(outputfile[1]), spider_image(outputfile[2]), 
                       spider_tuple(error_limit, chi2_limit), spider_tuple(iter_limit, reg_mode), spider_tuple(lambda_weight))
        return outputfile
    
    def ce_fit(session, inputfile, reference, mask, outputfile=None, **extra):
        '''Finds the linear transformation (applied to pixels) which 
         fits the histogram of the image file to the histogram of the reference file.
                
        `Original Spider (CE FIT) <http://www.wadsworth.org/spider_doc/spider/docs/man/cefit.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        reference : str
                    Reference input file
        mask : str
               Mask to use for image correction
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        _logger.debug("Enhance contrast of the image")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        session.invoke('ce fit', spider_image(reference), spider_image(inputfile), spider_image(mask), spider_image(outputfile))
        return outputfile
    
    def cg_ph(session, inputfile, **extra):
        ''' Compute center of gravity of image/volume using phase approximation.
                
        `Original Spider (CG PH) <http://www.wadsworth.org/spider_doc/spider/docs/man/cgph.html>`_
        
        ..todo:: determine if image or volume (assumes volume)
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        extra : dict
                Unused key word arguments
        
        :Returns:
            
        x : int
            Center of gravity x-coordinate
        y : int
            Center of gravity y-coordinate
        z : int
            Center of gravity z-coordinate (For volume only)
        rx : float
            Center of gravity x-coordinate
        ry : float
            Center of gravity y-coordinate
        rz : float
            Center of gravity z-coordinate (For volume only)
        '''
        
        _logger.debug("Determining the center of gravity (phase apporximation) for input image")
        session.invoke('cg ph x21,x22,x23,x24,x25,x26', spider_image(inputfile))
        return (session['x21'], session['x22'], session['x23'], session['x24'], session['x25'], session['x26'])
    
    def cp(session, inputfile, outputfile=None, **extra):
        '''Make a copy of a SPIDER image file
        
        `Original Spider (CP) <http://www.wadsworth.org/spider_doc/spider/docs/man/cp.html>`_
        
        .. todo:: uses is_spider_stack
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        
        _logger.debug("Copy a SPIDER image file")
        inputfile = session.replace_ext(inputfile)
        istack, = session.fi_h(inputfile, ('ISTACK', ))
        if not isinstance(inputfile, tuple) and istack > 0:
            stack_count,  = session.fi_h(spider_stack(inputfile), ('MAXIM', ))
            if outputfile is None: outputfile = session.ms(stack_count, spider_stack( (inputfile, 1) ))
            session.invoke('cp', spider_stack(inputfile, stack_count), spider_select(int(stack_count)), spider_stack(outputfile, stack_count), spider_select(int(stack_count)))
        else:
            if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
            session.invoke('cp', spider_image(inputfile), spider_image(outputfile))
        return outputfile

    def cp_from_mrc(session, inputfile, outputfile=None, **extra):
        '''Make a copy of a SPIDER image file in MRC format
        
        `Original Spider (CP FROM MRC) <http://www.wadsworth.org/spider_doc/spider/docs/man/cpfrommrc.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'cp from mrc', inputfile, outputfile, "Copy a SPIDER image file from MRC")
    
    def cp_to_mrc(session, inputfile, outputfile=None, **extra):
        '''Make a copy of a SPIDER image file in MRC format
        
        `Original Spider (CP TO MRC) <http://www.wadsworth.org/spider_doc/spider/docs/man/cptomrc.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'cp to mrc8', inputfile, outputfile, "Copy a SPIDER image file to MRC", spider_tuple(-9999))
    
    def cp_to_tiff(session, inputfile, outputfile=None, **extra):
        '''Copies a SPIDER file to a Tiff format file.
                
        `Original Spider (CP TO TIFF) <http://www.wadsworth.org/spider_doc/spider/docs/man/cptotiff.html>`_
        
        .. todo :: Add support for volume slice number
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'cp to tiff', inputfile, outputfile, "Copy a SPIDER image file")
    
    def dc_s(session, inputfile, bin_fac, outputfile=None, **extra):
        ''' Shrink a micrograph using the Spider decimation algorithm
                
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        bin_fac : float
                     Reduce the image size by this factor
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        if isinstance(bin_fac, int):
            z_size,  = session.fi_h(inputfile, ('NSLICE'))
            if z_size > 1: bin_fac = (bin_fac, bin_fac, bin_fac)
            else: bin_fac = (bin_fac, bin_fac)
        return spider_session.spider_command_fifo(session, 'dc s', inputfile, outputfile, "Decimating micrograph", spider_tuple(*bin_fac))
    
    def de(session, inputfile, **extra):
        '''Delete a file
        
        alias: delete_file
        
        `Original Spider (DE) <http://www.wadsworth.org/spider_doc/spider/docs/man/de.html>`_
        
        .. note::
        
            No way to test if this failed
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    File path to input image or volume
        extra : dict
                Unused key word arguments
        '''
        
        if is_incore_filename(inputfile) and hasattr(inputfile, 'hook'): 
            inputfile.hook = None
        _logger.debug("Delete file: %s"%str(inputfile))
        session.invoke('de', spider_image(inputfile))
    
    def du(session, inputfile, du_nstd, du_type=3, **extra):
        '''Eliminates all data in a picture that is more than a given multiple 
        of the standard deviation away from the mode of the histogram. The 
        eliminated data are set to the boundaries of the range.
                
        `Original Spider (DU) <http://www.wadsworth.org/spider_doc/spider/docs/man/du.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        du_nstd : int
                  Number of standard deviations
        du_type : int
                  Dedusting type: (1) BOTTOM, (2) TOP, (3) BOTH SIDES: 3
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        inputfile : str
                    Input filename
        '''
        
        _logger.debug("Get value of pixel in an image")
        if du_type not in (1, 2, 3): raise spider_session.SpiderParameterError, "du_type must be 1, 2 or 3"
        session.invoke('du', spider_image(inputfile), spider_tuple(du_nstd), spider_tuple(du_type))
        return inputfile
    
    def fd(session, inputfile, scatterfile, outputfile=None, **extra):
        '''Applies Fourier filter to 2D or 3D to real or Fourier image. Coefficients of the filter are read from a document file.
        
        `Original Spider (FD) <http://www.wadsworth.org/spider_doc/spider/docs/man/fd.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        scatterfile : str
                      Filter document file
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'fd', inputfile, outputfile, "Filtering image using a filter built from a file", spider_image(scatterfile))
    
    def fi_h(session, inputfile, header, **extra):
        '''Retrieve particular values from the file header by 
        name and optionally place variable values in specified register variables.
        
        .. note::
            
            #. `header` can be either a list (of strings) or a string
        
        `Original Spider (FI H) <http://www.wadsworth.org/spider_doc/spider/docs/man/fih.html>`_
        
        .. todo:: Need to test if values are valid when given to the header
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        header : str
                 Comma separated list of headers to query
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        values : tuple
                 Namedtuple of values where fields correspond to header names
        '''
        
        _logger.debug("Querying information from file header")
        if isinstance(header, str): header = header.split(',')
        hreg = ["[x%d]"%(i+1) for i in xrange(len(header))]
        session.invoke('fi h %s'%(",".join(hreg)), spider_image(inputfile), ",".join(header))
        if int(session[9]) > 0: raise spider_session.SpiderCommandError, "fi h failed to inquire header from the given file"
        return collections.namedtuple("fih", ",".join(header))._make([session[r] for r in hreg])
    
    LP, HP, GAUS_LP, GAUS_HP, FERMI_LP, FERMI_HP, BUTER_LP, BUTER_HP = range(1, 9)
    def fq(session, inputfile, filter_type=7, filter_radius=0.12, pass_band=0.1, stop_band=0.2, temperature=0.3, outputfile=None, **extra):
        ''' Applies Fourier filters to 2-D or 3-D images. Images need not have power-of-two dimensions. Padding with the average is applied during filtration.
                
        `Original Spider (FQ) <http://www.wadsworth.org/spider_doc/spider/docs/man/fq.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        filter_type : int
                      Type of filter:
                          #. LOW-PASS
                          #. HIGH-PASS
                          #. GAUSS LOW-PASS
                          #. GAUSS HIGH-PASS
                          #. FERMI LOW-PASS
                          #. FERMI HIGH-PASS
                          #. BUTER. LOW-PASS
                          #. BUTER. HIGH-PASS
        filter_radius : float
                        The FILTER RADIUS can be given either in absolute units or pixel units. 
                        If answer is > 1.0 it is treated as given in pixel units. If filter function 
                        radius is given in frequency units, they should be in the range 0.0<=f<=0.5
                        (All except, Butterworth)
        pass_band : float
                    Allowed frequence cutoff for band-pass filter (Butterworth type)
        stop_band : float
                    Disallowed frequence cutoff for band-pass filter (Butterworth type)
        temperature : float
                      Roughly within this reciprocal distance for filter fall off (Fermi type)
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        filter_type = int(filter_type)
        if filter_type < 1 or filter_type > 8: raise ValueError, "Filter type must be an integer between 1-8"
        args = []
        if filter_type < 7: args.append( spider_tuple(filter_radius) )
        if filter_type in (5, 6): args.append( spider_tuple(temperature) )
        if filter_type in (7, 8): args.append( spider_tuple(pass_band, stop_band) )
        return spider_session.spider_command_fifo(session, 'fq', inputfile, outputfile, "Applying the Filter to the image or volume", spider_tuple(filter_type), *args)
    
    def fs(session, inputfile, **extra):
        ''' To compute and list statistical parameters 
        (i.e. minimum, maximum, average, and standard deviation) 
        of an image/volume.
                
        `Original Spider (FS) <http://www.wadsworth.org/spider_doc/spider/docs/man/fs.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        maximum : float
                  Maximum of values in the image
        minimum : float
                  Minimum of values in the image
        mean : float
               Mean of values in the image
        std : float
              Standard deviation of values in the image
        '''
        
        _logger.debug("Calculate statistics of an image")
        session.invoke('fs x11,x12,x13,x14', spider_image(inputfile))
        return session['x11'], session['x12'], session['x13'], session['x14']
    
    def ft(session, inputfile, outputfile=None, **extra):
        ''' Computes forward Fourier transform of a 2D or 3D image, or inverse Fourier 
        transform of a complex Fourier-formatted file
                
        `Original Spider (FT) <http://www.wadsworth.org/spider_doc/spider/docs/man/ft.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        return spider_session.spider_command_fifo(session, 'ft', inputfile, outputfile, "Applying the fourier transform to an image")
    
    def gp(session, inputfile, location, **extra):
        '''Gets pixel value from specified location in image/volume.
                
        `Original Spider (GP) <http://www.wadsworth.org/spider_doc/spider/docs/man/gp.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        location : tuple
                   2-tuple or 3-tuple depending on image dimensions
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        value : float
                Pixel value
        '''
        
        _logger.debug("Get value of pixel in an image")
        session.invoke('gp x11', spider_image(inputfile), spider_tuple(*location))
        return session['x11']
    
    def ip(session, inputfile, size, outputfile=None, **extra):
        '''Takes input image/volume of any dimension and creates interpolated image/volume 
        of any dimension. Uses bilinear interpolation for images and trilinear interpolation on volumes.
        
        `Original Spider (IP) <http://www.wadsworth.org/spider_doc/spider/docs/man/ip.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        size : tuple
               New image size
        outputfile : str
                     Filename for output file
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename for output file
        '''
        
        return spider_session.spider_command_fifo(session, 'ip', inputfile, outputfile, "Resize image/volume with interpolation", spider_tuple(*size))
    
    def ip_fs(session, inputfile, size, outputfile=None, **extra):
        ''' Resize image/volume with bicubic spline || interpolation
        
        `Original Spider (IP FS) <http://www.wadsworth.org/spider_doc/spider/docs/man/iqfi.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        size : tuple
               New image size
        outputfile : str
                     Filename for output file
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename for output file
        '''

        return spider_session.spider_command_fifo(session, 'ip fs', inputfile, outputfile, "Resize image/volume with interpolation", spider_tuple(*size))
    
    def ip_ft(session, inputfile, size, outputfile=None, **extra):
        '''Takes input image/volume of any dimension and creates interpolated image/volume 
        of any dimension. Creates enlarged image/volume using zero padding in Fourier Space.
        
        `Original Spider (IP FT) <http://www.wadsworth.org/spider_doc/spider/docs/man/ipft.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        size : tuple
               New image size
        outputfile : str
                     Filename for output file
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename for output file
        '''
        
        return spider_session.spider_command_fifo(session, 'ip ft', inputfile, outputfile, "Resize image/volume with interpolation", spider_tuple(*size))
    
    def iq_fi(session, inputfile, **extra):
        '''To inquire whether a file exists or not
                
        `Original Spider (IQ FI) <http://www.wadsworth.org/spider_doc/spider/docs/man/iqfi.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        exists : bool
                 True if file exists
        '''
        
        _logger.debug("Querying if file exists")
        session.invoke('iq fi [x1]', spider_image(inputfile))
        return int(session['x1'])
    
    def iq_sync(session, inputfile, check_delay=60, total_delay=600, **extra):
        '''To wait until a file exists. A primitive method of synchronizing different SPIDER runs.
        
        `Original Spider (IQ SYNC) <http://www.wadsworth.org/spider_doc/spider/docs/man/iqsync.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        check_delay : int
                      Seconds for delay between checks
        total_delay : int
                      Total delay before giving up
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        exists : int
                 Number of seconds waited
        '''
        
        _logger.debug("Wait for a file to exist")
        session.invoke('iq sync [x1]', spider_image(inputfile), spider_tuple(check_delay, total_delay))
        return int(session['x1'])
    
    def li_d(session, inputfile, info_type='W', row=56, column=34, header_pos=(1,2,3), use_2d=True, use_phase=False, outputfile=None, **extra):
        '''Lists specified elements of a given file in document file.
                
        `Original Spider (LI D) <http://www.wadsworth.org/spider_doc/spider/docs/man/lid.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        info_type : str
                    Type of information to list: .HEADER, PIXEL, ROW, COLUMN, IMAGE, OR WINDOW (H/P/R/C/I/W)
        row : tuple
              Row or tuple describing a row range
        column : tuple
                 Column or tuple describing a column range
        header_pos : tuple
                     Tuple or list of header positions
        use_2d : bool
                 Set false for a 3D image
        use_phase : bool
                    Set true for  phase/modulus listing
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        _logger.debug("List values of an image to a document file")
        if outputfile is None: raise ValueError, "Incore documents curenttly not supported by LI D"
        if info_type not in ('H', 'P', 'R', 'C', 'I', 'W'): 
            raise spider_session.SpiderParameterError, "Info type must be .HEADER, PIXEL, ROW, COLUMN, IMAGE, OR WINDOW (H/P/R/C/I/W)"
        additional = []
        if   info_type == 'H': additional.append( spider_tuple(*header_pos) )
        elif info_type == 'P': additional.append( spider_tuple(column, row) )
        elif info_type == 'R': additional.append( '(%d)'%int(row) ) #spider_select(row) )
        elif info_type == 'C': additional.append( spider_select(column) )
        elif info_type == 'I': additional.append( spider_tuple(row) )
        elif info_type == 'W': additional.extend( [spider_select(column), spider_select(row)] )
        if use_2d:
            z_size,  = session.fi_h(inputfile, ('NSLICE'))
            if z_size > 1: use_2d = False
        session.invoke('li d', spider_image(inputfile), spider_doc(outputfile), info_type, *additional)
        return outputfile
    
    def ma(session, inputfile, radius, center, mask_type='D', background_type='P', width=3.5, half_width=0.0, background=3.0, outputfile=None, **extra):
        '''Masks a specified picture with circular masks of specified radii. Pixels 
        in the area inside the inner circle and the area outside the outer circle 
        are set to a specified background.
        
        `Original Spider (MA) <http://www.wadsworth.org/spider_doc/spider/docs/man/ma.html>`_
        
        .. todo:: Check order of parameters
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        radius : tuple
                 Radius of the disk (if tuple, outer and inner radius)
        center : tuple
                 Center coordinates either 2 or 3-tuple
        mask_type : str
                    Type of the mask: (D)isk, (C)osine, (G)aussian edge, or (T)rue Gaussian
        background_type : str
                          Type of the background (A)V, (P)REC AV, (C)IRCUMF, OR (E)XTERNAL
        width : float
                Width of the Cosine
        half_width : float
                     Half-width of the Gaussian
        background : float
                     Backgroud pixel value
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        if mask_type not in ('D', 'C', 'G', 'T'): raise spider_session.SpiderParameterError, "mask type must be D/C/G/T"
        if background_type not in ('A', 'P', 'C', 'E'): raise spider_session.SpiderParameterError, "mask type must be A/P/C/E"
        if not isinstance(radius, tuple): radius = (radius, 0)
        elif len(radius) == 1: radius = ( int(radius[0]), 0)
        additional = [ spider_tuple(int(center[0]), int(center[1])) ]
        if len(center) > 2: additional.append( spider_tuple(int(center[2])) )
        if half_width == 0.0: half_width = width
        if mask_type == 'C': additional.append(spider_tuple(width))
        elif mask_type == 'G': additional.append(spider_tuple(half_width))
        if background_type == 'E': additional.append(spider_tuple(background))
        return spider_session.spider_command_fifo(session, 'ma', inputfile, outputfile, "Mask an image", spider_tuple(*radius), mask_type, background_type, *additional)
    
    def md(session, mode, value=None, **extra):
        ''' Switches between different operating modes of SPIDER or sets certain options.
        
        `Original Spider (MD) <http://www.wadsworth.org/spider_doc/spider/docs/man/md.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        mode : str
               Specific mode to change
        value : str
                New value of the mode
        extra : dict
                Unused key word arguments
        '''
        
        additional = []
        if mode == 'SET MP': additional.append(spider_tuple(value))
        else: raise ValueError, "Only SET MP supported"
        session.invoke('md', mode, *additional)
    
    def mo(session, image_size, model="T", background_constant=12.0, circle_radius=12, gaus_center=(12.0, 12.0), gaus_std=4.2, gaus_mean=1.0, rand_gauss=False, outputfile=None, **extra):
        ''' Creates a model image. The following options are available: 
        (B)LANK -- BLANK IMAGE 
        (C)IRCLE -- FILLED CIRCLE 
        (G)AUSSIAN -- GASUSIAN DENSITY DISTRIBUTION 
        (R)ANDOM -- RANDOM UNIFORM/GAUSSIAN STATISTICS 
        (S)INE -- SET OF SINE WAVES  -----------------  (NOT SUPPORTED)
        (T)EST -- 2D SINE WAVE 
        (W)EDGE -- DENSITY WEDGE
                    
        .. todo :: 1. Support sine waves
        
        `Original Spider (MO) <http://www.wadsworth.org/spider_doc/spider/docs/man/mo.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        image_size : (int,int)
                     Size of the image
        model : str
                Type of the model object
        background_constant : float
                              Background for (B)lank image
        circle_radius : int
                        Radius for filled (C)ircle 
        gaus_center : (float,float)
                      Center for (G)aussian distribution
        gaus_std : float
                   Standard deviation for (G)aussian and (R)andom distributions
        gaus_mean : float
                    Mean for (R)andom gaussian distribution
        rand_gauss : bool
                     True for (R)andom gaussian distribution
        outputfile : str
                 Filename of the output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Int or str describing the output file
        '''
        
        _logger.debug("Creating model image of size %s in file %s: "%(str(image_size), str(outputfile)))
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        model = model.upper()
        if model not in ('B', 'C', 'G', 'R', 'T', 'W'): raise ValueError, "Only B/C/G/R/S/T/W supported for model"
        additional = []
        if model == 'B':
            additional.append(spider_tuple(background_constant))
        elif model == 'C':
            additional.append(spider_tuple(circle_radius))
        elif model == 'G':
            additional.extend([spider_tuple(*gaus_center), spider_tuple(gaus_std)])
        elif model == 'R':
            if rand_gauss: additional.extend([spider_tuple('Y'), spider_tuple(gaus_mean, gaus_std)])
            else: additional.append(spider_tuple('N'))
        session.invoke('mo', spider_image(outputfile), spider_tuple(*image_size), spider_tuple(model), *additional)
        return outputfile
    
    def mr(session, inputfile, axis='Y', outputfile=None, **extra):
        '''Creates mirror-symmetry related output image from input image, 
        with the mirror axis lying at row number NROW/2+1, or NSAM/2+1, 
        or NSLICE/2+1. Works for 2D and 3D files.
                
        `Original Spider (MR) <http://www.wadsworth.org/spider_doc/spider/docs/man/mr.html>`_
        
        .. note::
            
            #. If image_width is given as a string, then it is assumed to be a filename - `fi_h` will be used to query both width and height for the created stack
            #. If height is zero, it will be set to the width
            #. A new incore file identifier is generated to workaround a spider bug
        
        .. todo:: Report bug - requires new incore file
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        axis : str
               Axis to mirror image over
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Name of the outputfile
        '''
        
        return spider_session.spider_command_fifo(session, 'mr', inputfile, outputfile, "Creates mirror-symmetry", axis)
    
    def ms(session, num_images, image_width=None, height=0, depth=1, outputfile=None, window=None, **extra):
        '''Creates an empty inline stack
                
        `Original Spider (MS) <http://www.wadsworth.org/spider_doc/spider/docs/man/ms.html>`_
        
        .. note::
            
            #. If image_width is given as a string, then it is assumed to be a filename - `fi_h` will be used to query both width and height for the created stack
            #. If height is zero, it will be set to the width
            #. A new incore file identifier is generated to workaround a spider bug
        
        .. todo:: Report bug - requires new incore file
        
        :Parameters:
        
        session : Session
                  Current spider session
        num_images : int
                     Number of images in the stack
        image_width : int
                      Width of the image
        height : int
                 Height of the image (Default: 0)
        depth : int
                 Depth of the image (Default: 1)
        outputfile : str
                     Filename for incore stack (Default: None)
        window : int
                 Window size from the params file
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Name of the outputfile
        '''
        
        _logger.debug("Create an inline stack - True")
        if not isinstance(num_images, int): raise spider_session.SpiderParameterError, "Number of images currently required to be an integer"
        if outputfile is None: outputfile = session.temp_incore_image(True, hook=session.de, is_stack=True)#(True)
        if image_width is None: image_width = window
        if isinstance(image_width, str) or isinstance(image_width, tuple): 
            vals = session.fi_h(image_width, ('NSAM', 'NROW'))
            image_width, height = int(vals.NSAM), int(vals.NROW)
            if height == 0 or image_width == 0: raise spider_session.SpiderCommandError, "Failed to query image dimensions"
        if isinstance(outputfile, tuple): outputfile = outputfile[0]
        if height == 0: height = image_width
        session.invoke('ms', spider_stack(outputfile), spider_tuple(image_width, height, depth), spider_tuple(num_images))
        return outputfile
    
    def mu(session, inputfile, *otherfiles, **extra):
        ''' Multiply a set of images
                
        `Original Spider (MU) <http://www.wadsworth.org/spider_doc/spider/docs/man/mu.html>`_
        
        .. note:: 
        
            With this command you must use outputfile="name of outputfile", 
            e.g. session.mu(inputfile, anotherfile1, anotherfile2, outputfile=outputfile)
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        otherfiles : str
                     Other filenames listed on the as function parameters
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        return spider_session.spider_command_multi_input(session, 'mu', "Multiply images", inputfile, *otherfiles, **extra)
    
    def neg_a(session, inputfile, outputfile=None, **extra):
        ''' Multiply a set of images
                
        `Original Spider (NEG A) <http://www.wadsworth.org/spider_doc/spider/docs/man/neg_a.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        return spider_session.spider_command_fifo(session, 'neg a', inputfile, outputfile, "Invert contrast")
    
    OR_SH_TUPLE = collections.namedtuple("orsh", "psi,x,y,mirror,cc")
    def or_sh(session, inputfile, reference, trans_range=6, trans_step=2, ring_first=2, ring_last=15, test_mirror=True, window=None, **extra):
        '''Determines rotational and translational orientation between two images after resampling into 
        polar coordinates with optional additional check of mirror transformation. This is the same 
        as: 'AP SH' except it only processes a single pair of images. 
                
        `Original Spider (OR SH) <http://www.wadsworth.org/spider_doc/spider/docs/man/orsh.html>`_
        
        .. note::
        
            #. If `window` (params file), the `ring_last` will be set to 1/2 window - `trans_range` - 3
            
        .. todo:: trans_range must be divisible by trans_step - increase if less than box size - ring radius
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of experimental input image
        reference : str
                    Filename of reference image
        trans_range : int
                      Translation range
        trans_step : int
                     Translation step size
        ring_first : int
                     Start of polar coordinate radial rings analyzed for rotational alignment
        ring_last : int
                     End of polar coordinate radial rings analyzed for rotational alignment
        test_mirror : bool
                      Test the mirror of the experimental image
        window : int
                 Size of the particle window
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        psi : float
              Angle of in plane rotation (PSI)
        x : float
            Translation in the x-direction
        y : float
            Translation in the y-direction
        mirror : bool
                 Translation in the x-direction
        cc : float
             Cross-correlation peak
        '''
        
        _logger.debug("Performing orientation search")
        if window is not None and (window/2 - ring_last - trans_range) < 2:
            old_ring_last = ring_last
            ring_last = window/2 - trans_range - 3
            _logger.debug("Overriding %d to %d for image dimension from PARAMS"%(ring_last, old_ring_last))
        session.invoke('or sh x21,x22,x23,x24,x25', spider_image(reference), spider_tuple(trans_range, trans_step), 
                       spider_tuple(ring_first, ring_last), spider_image(inputfile), spider_tuple(test_mirror))
        return session.OR_SH_TUPLE(session['x21'], session['x22'], session['x23'], session['x24'], session['x25'])
    
    def pd(session, inputfile, window_size, center_coord=None, background='Y', background_value=0.0, outputfile=None, **extra):
        ''' To pad an image/volume to make a larger image/volume. Places the input image at specified 
        position and pads the input with a specified background value.
        
        `Original Spider (PD) <http://www.wadsworth.org/spider_doc/spider/docs/man/pd.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        window_size : tuple
                      New size of the image/volume window
        center_coord : tuple
                      New size of the image/volume window
        background : str
                     Type of background: Y - overall average, N - specified value, B - average density of perimeter, M - minimum density
        background_value : float
                           Specified background value
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        z_size, x_size, y_size, = session.fi_h(inputfile, ('NSLICE', 'NSAM', 'NROW'))
        if z_size > 1:
            if not hasattr(window_size, '__len__'): window_size = (int(window_size), int(window_size), int(window_size))
            elif len(window_size) == 1: window_size = (int(window_size[0]), int(window_size[0]), int(window_size[0]))
            elif len(window_size) == 2: window_size = (int(window_size[0]), int(window_size[0]), int(window_size[0]))
        else:
            if not hasattr(window_size, '__len__'): window_size = (window_size, window_size, 1)
            elif len(window_size) == 1: window_size = (int(window_size[0]), int(window_size[0]), 1)
            elif len(window_size) == 2: window_size = (int(window_size[0]), int(window_size[0]), 1)
        if center_coord is None:
            if z_size > 1:
                center_coord = (int(window_size[0]-x_size)/2+1, int(window_size[1]-y_size)/2+1, int(window_size[0]-z_size)/2+1) #([padsize]-[winsiz])/2+1,([padsize]-[winsiz])/2+1
            else:
                center_coord = (int(window_size[0]-x_size)/2+1, int(window_size[1]-y_size)/2+1)
        else:
            if z_size > 1:
                if not hasattr(center_coord, '__len__'): center_coord = (center_coord, center_coord, center_coord)
                elif len(center_coord) == 1: center_coord = (center_coord[0], center_coord[0], center_coord[0])
                elif len(center_coord) == 2: center_coord = (center_coord[0], center_coord[1], center_coord[0])
            else:
                if isinstance(center_coord, int): center_coord = (center_coord, center_coord, 1)
                elif len(center_coord) == 1: center_coord = (center_coord[0], center_coord[0], 1)
                elif len(center_coord) == 2: center_coord = (center_coord[0], center_coord[1], 1)
        
        additional=[]
        if background == 'N': additional.append( spider_tuple(background_value) )
        return spider_session.spider_command_fifo(session, 'pd', inputfile, outputfile, "Pad images", spider_tuple(*window_size), background, spider_tuple(*center_coord), *additional)
    
    def pj_3q(session, inputfile, angle_doc, angle_list, pj_radius=-1, pixel_diameter=None, outputfile=None, **extra):
        '''Computes projection(s) of a 3D volume according to the three Eulerian angles.
        
        alias: project_3d
        
        `Original Spider (PJ 3Q) <http://www.wadsworth.org/spider_doc/spider/docs/man/pj3q.html>`_
        
        .. note::
            
            #. This command automatically calls `ms` or make stack when the outputfile is None, i.e. when an incore file is automatically used.
            #. If the `pj_radius` is 0 or less, this command sets the pj_radius to 0.69*pixel_diameter where pixel_diameter is defined in the params file
            #. Currently, `angle_list` only supports a single integer, not a selection file
            #. If the command fails and no file is generated, it will raise an exception, spider_session.SpiderCommandError
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for 3D volume
        angle_doc : str
                    Document file containing the angles
        angle_list : str
                     List of angles numbers either file or range (1-[numang])
        pj_radius : int
                    Radius of sphere to compute projection, if less than one use 0.69 times the diameter of the object in pixels (Default: -1)
        pixel_diameter : int
                         Pixel diameter of the particle
        outputfile : str
                    Filename to store 2D projections (If none, temporary incore file is used and returned)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                    Tuple containing the output file and number of angles
        '''
        
        _logger.debug("Create 2D projections of a 3D object")
        angle_list, max_count, total_size = spider_session.ensure_stack_select(session, None, angle_list)
        if outputfile is None: outputfile = session.ms(total_size, spider_image(inputfile))
        elif int(session.fi_h(spider_stack(inputfile), 'NSAM')[0]) != int(session.fi_h(spider_stack(outputfile), 'NSAM')[0]):
            session.de(outputfile)
            outputfile = session.ms(total_size, spider_image(inputfile), outputfile=outputfile)
        #param['reference_stack'] = spi.ms(max_ref_proj, param['window'])
        
        if pj_radius is None or pj_radius < 1:
            if pixel_diameter is None: raise spider_session.SpiderParameterError, "Either radius or pixel_diameter must be set"
            pj_radius = 0.69 * pixel_diameter
        session.invoke('pj 3q', spider_image(inputfile), spider_tuple(pj_radius), spider_select(angle_list), spider_doc(angle_doc), spider_stack(outputfile, max_count))
        return outputfile
    
    def pw(session, inputfile, outputfile=None, **extra):
        '''Generates full, unscrambled Fourier moduli from complex 
        Fourier transform for 2-D or 3-D pictures. The input image 
        can be real if it fits into the memory. Operation supports 
        multiple radix FFT, thus the input image does not have to 
        have power-of-two dimensions. The resulting data are real 
        and can be displayed like a normal picture.
                
        `Original Spider (PW) <http://www.wadsworth.org/spider_doc/spider/docs/man/pw.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename (Fourier Transform)
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'pw', inputfile, outputfile, "Estimate power spectrum of an image")
    
    def ra(session, inputfile, outputfile=None, **extra):
        '''Fits a least-squares plane to the picture, and subtracts the plane from the 
           picture. A wedge-shaped overall density profile can thus be removed from the picture.
                
        `Original Spider (RA) <http://www.wadsworth.org/spider_doc/spider/docs/man/ra.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input file template, stack template or stack
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'ra', inputfile, outputfile, "Estimate and remove ramp from image")
    
    def rb_32f(session, inputfile, angle_file, input_select=None, sym_file="", outputfile=None, **extra):
        '''Changes the scale, rotates, and shifts image circularly. Then calculates two 
        randomly selected sample recontruction and a total-3D reconstruction using 
        interpolation in Fourier space.Rotates counter-clockwise around the center (NSAM/2 + 1, NROW/2 + 1). 
        Negative angles = clockwise. Note that the terms "clockwise" and "counter-clockwise" refer to the mirrored 
        x-y system used for image display). This operation is the same as 'RT SQ' followed by 'BP 32F. It is 
        about 5%-10% faster than that operation sequence.
                
        `Original Spider (RB 32F) <http://www.wadsworth.org/spider_doc/spider/docs/man/rb32f.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input file template, stack template or stack
        angle_file : str
                     File containing euler angles and shifts
        input_select : int
                       Input filename or number of images in the stack
        sym_file : str
                   Input filename with symetries
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output full volume
        outputfile1 : str
                      Output half volume
        outputfile2 : str
                      Output half volume
        '''
        
        _logger.debug("Back project")
        outputfile = spider_session.ensure_output_recon3(session, outputfile)
        input_select, max_count = spider_session.ensure_stack_select(session, inputfile, input_select)[:2]
        session.invoke('rb 32f', spider_stack(inputfile, max_count), spider_select(input_select), spider_doc(angle_file), spider_doc(sym_file), spider_doc(None), spider_image(outputfile[0]), spider_image(outputfile[1]), spider_image(outputfile[2]))
        return outputfile
    
    def rf_3(session, inputfile1, inputfile2, ring_width=0.5, lower_scale=0.2, upper_scale=2.0, missing_ang='C', max_tilt=90.0, noise_factor=3.0, outputfile=None, **extra):
        '''Calculate the differential 3-D phase residual and the Fourier Shell Correlation between 
        two volumes. The Differential Phase Residual over a shell with thickness given by shell width 
        and the Fourier Shell Correlation between shells of specified widths are computed and stored 
        in the document file. Does not need powers of two dimensions (for exclusions see 'FT' operation) 
        and takes real or Fourier input volumes. NSAM, NROW and NSLICE need to be the same in both volumes.
                
        `Original Spider (RF 3) <http://www.wadsworth.org/spider_doc/spider/docs/man/rf3.html>`_
        
        .. todo:: Supports incore document files (really?)
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile1 : str
                     Input file containing first half volume
        inputfile2 : str
                     Input file containing second half volume
        ring_width : float
                     Shell thickness in reciprocal space sampling units
        lower_scale : float
                      Lower range of scale factors by which the second Fourier must be multiplied for the comparison
        upper_scale : float
                      Upper range of scale factors by which the second Fourier must be multiplied for the comparison
        missing_ang : str
                      `C` if you have a missing cone and `W` if you have a missing wedge
        max_tilt : float
                   Angle of maximum tilt angle in degrees
        noise_factor : float
                       Factor given here determines the FSCCRIT. Here 3.0 corresponds to the 3 sigma criterion i.e., 3/SQRT(N), 
                       where N is number of voxels for a given shell.
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        values : tuple
                 Namedtuple of values where fields correspond to header names
        '''
        
        _logger.debug("Create an inline stack")
        if outputfile is None: 
            width, = session.fi_h(inputfile1, ('NSAM'))
            outputfile = session.sd_ic_new((5, width+1))
        session.invoke('rf 3 [x1],[x2]', spider_image(inputfile1), spider_image(inputfile2), spider_tuple(ring_width), spider_tuple(lower_scale, upper_scale), missing_ang, spider_tuple(max_tilt), spider_tuple(noise_factor), spider_doc(outputfile))
        return outputfile, float(session['x1']), float(session['x2'])
    
    def ro(session, inputfile, outputfile=None, **extra):
        '''Computes the radial distribution function of a two or three dimensional density distribution stored 
        in a square array. Center assumed to be located at (NSAM/2+1, NROW/2+1).
                
        `Original Spider (RO) <http://www.wadsworth.org/spider_doc/spider/docs/man/ro.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'ro', inputfile, outputfile, "Take a rotational average of an image")
    
    def rtd_sq(session, inputfile, alignment, select=None, alignment_cols=(6,0,7,8), outputsel=None, outputfile=None, **extra):
        '''Changes the scale, rotates, and shifts image circularly. Rotates counter-clockwise 
        around the center (NSAM/2 + 1, NROW/2 + 1). (Negative angles = clockwise. Note that the 
        terms "clockwise" and "counter-clockwise" refer to the mirrored x-y system used for 
        image display) Output image has SAME size as input image.
                
        `Original Spider (RTD SQ) <http://www.wadsworth.org/spider_doc/spider/docs/man/rtdsq.html>`_
        
        .. todo :: 
        
            - Set of stacked images unsupported
            - Set alignment parameres with cols for single image
            - fix this command
        
        :Parameters:
            
            session : Session
                      Current spider session
            inputfile : str
                        Filename of input image projection stack
            alignment : str
                        Filename for the alignment parameters or for single image, tuple of alignment parameters
            select : str
                     Selection file
            alignment_cols : tuple
                             List of alignment columns
            outputsel : str
                        Output selection file
            outputfile : str
                         Filename of output image (If none, temporary incore file is used and returned)
            extra : dict
                    Unused key word arguments
        
        :Returns:
            
            outputfile : str
                         Filename of output image
        '''
        
        _logger.debug("Performing rotation and translation")
        if outputfile is None:
            if (isinstance(inputfile, tuple) or is_incore_filename(inputfile) or inputfile.find('@') == -1) and select is None:
                outputfile = session.temp_incore_image(hook=session.de)
            else:
                if select is not None:
                    if len(select) == 3: nrows = select[2]
                    else: nrows = session.ud_n(select[0])[2]
                else:
                    nrows = session.ud_n(alignment)[2]
                _logger.debug("Using ms to create in core file")
                outputfile = session.ms(nrows, spider_stack( (inputfile, 1) ))
        
        stack_total, = session.fi_h(spider_stack(inputfile), ('MAXIM'))
        session.invoke('rtd sq', spider_stack(inputfile, stack_total), spider_select(select), spider_tuple(*alignment_cols), spider_doc(alignment), spider_stack(outputfile, stack_total), spider_select(outputsel))
        return outputfile
    
    def rt_sq(session, inputfile, alignment, input_select=None, alignment_cols=(6,0,7,8), outputfile=None, **extra):
        '''Changes the scale, rotates, and shifts image circularly. Rotates counter-clockwise 
        around the center (NSAM/2 + 1, NROW/2 + 1). (Negative angles = clockwise. Note that the 
        terms "clockwise" and "counter-clockwise" refer to the mirrored x-y system used for 
        image display) Output image has SAME size as input image.
                
        `Original Spider (RT SQ) <http://www.wadsworth.org/spider_doc/spider/docs/man/rtsq.html>`_
        
        .. todo :: 
        
            - Set of stacked images unsupported
            - Single image unsupported
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image projection stack
        alignment : str
                    Filename for the alignment parameters or for single image, tuple of alignment parameters
        input_select : str
                       Input selection file
        alignment_cols : tuple
                         List of alignment columns
        outputfile : str
                     Filename of output image (If none, temporary incore file is used and returned)
        extra : dict
                Unused key word arguments
        
        :Returns:
            
            outputfile : str
                         Filename of output image
        '''
        
        input_select, max_count, count = spider_session.ensure_stack_select(session, inputfile, input_select)
        assert(count > 1)
        if outputfile is None: outputfile = session.ms(count, spider_stack( (inputfile, 1) ))
        session.invoke('rt sq', spider_stack(inputfile, max_count), spider_select(input_select), spider_tuple(*alignment_cols), spider_doc(alignment), spider_stack(outputfile, max_count))
        return outputfile
        
    def sd_e(session, outputfile, **extra):
        '''Close an output document file
                
        `Original Spider (SD E) <http://www.wadsworth.org/spider_doc/spider/docs/man/sde.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     SPIDER reference to in core document file
        '''
        
        session.invoke('sd e', spider_doc(outputfile))
        return outputfile
    
    def sd_ic(session, key, values, outputfile, hreg=None, **extra):
        '''Create an incore doc file (a matrix in SPIDER)
                
        `Original Spider (SD IC NEW) <http://www.wadsworth.org/spider_doc/spider/docs/man/sdicnew.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        key : int
              Row of the array
        values : ndarray
                 List of fields to store in the array
        outputfile : str
                     Filename for incore stack (Default: None)
        hreg : list
               List of registers
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     SPIDER reference to in core document file
        '''
        
        if hreg is None: 
            if not hasattr(values, '__iter__'): hreg = ["[x1]"]
            else: hreg = ["[x%d]"%(i+1) for i in xrange(len(values))]
        if not hasattr(values, '__iter__'): 
            session[hreg[0]] = values
        else: 
            for i in xrange(len(hreg)): 
                session[hreg[i]] = values[i]
        session.invoke('sd ic %d,%s'%(key, ",".join(hreg)), spider_doc(outputfile))
        return outputfile
    
    def sd_ic_new(session, shape, outputfile=None, **extra):
        '''Create an incore doc file (a matrix in SPIDER)
                
        `Original Spider (SD IC NEW) <http://www.wadsworth.org/spider_doc/spider/docs/man/sdicnew.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        shape : tuple
                Number of columns and rows of the matrix
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     SPIDER reference to in core document file
        '''
        
        if outputfile is None:  outputfile = session.temp_incore_doc(hook=session.ud_ice)
        session.invoke('sd ic new', spider_doc(outputfile), spider_tuple(*shape))
        return outputfile
    
    def sh_f(session, inputfile, coords, outputfile=None, **extra):
        ''' Shifts a picture or a volume by a specified vector using Fourier interpolation.
        
        `Original Spider (SH F) <http://www.wadsworth.org/spider_doc/spider/docs/man/shf.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename of input image
        coords : tuple
                 Shift in x, y and z
        outputfile : str
                     Filename of output image (Default: None | e.g. create an incore file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Filename of output image
        '''
        
        return spider_session.spider_command_fifo(session, 'sh f', inputfile, outputfile, "Shifting a volume", spider_tuple(*coords))
    
    def sq(session, inputfile, outputfile=None, **extra):
        '''Square an image
        
        `Original Spider (SQ) <http://www.wadsworth.org/spider_doc/spider/docs/man/sq.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'sq', inputfile, outputfile, "Square an image")
    
    def tf_c3(session, defocus, cs, window=None, source=None, defocus_spread=None, ampcont=None, envelope_half_width=None, outputfile=None, **extra):
        '''To compute the phase contrast transfer function for bright-field 
        electron microscopy. For literature, see Notes. 'TF C3' produces 
        the transfer function in complex 3D form. It can then be applied, by 
        using 'MU' or 'TF CTS' , to the Fourier transform of a model object for 
        simulations of bright-field weak phase contrast.
                
        `Original Spider (TF C3) <http://www.wadsworth.org/spider_doc/spider/docs/man/tfc3.html>`_
        
        .. note::
            
            #. Except for `defocus`, all other parameters come directly from the SPIDER params file
            #. Electron wavelength is calculated from the microscope `voltage`
            #. Maximum spatial frequency is calculated from the angstrom per pixel
        
        .. note::
        
            This command is not intended to be used with `spider.setup_options`, 
            use `spider_params.read_parameters_to_dict` to setup the required parameters.
        
        .. seealso:: spider_session.generate_ctf_param
        
            For all required CTF parameters
        
        :Parameters:
        
        session : Session
                  Current spider session
        defocus : float
                  Amount of defocus, in Angstroems
        cs : float
             Spherical aberration constant
        window : int
                 Dimension of the 2D array
        source : float
                 Size of the illumination source in reciprocal Angstroems
        defocus_spread : float
                         Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
        ampcont : float
                  Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
        envelope_half_width : float
                              Envelope parameter specifies the 2 sigma level of the Gaussian
        outputfile : str
                    Filename to store the computed function (If none, temporary incore file is used and returned) (Defaut: None)
        extra : dict
                Unused key word arguments

        :Returns:
        
        outputfile : str
                    Output image of CTF model
        '''

        _logger.debug("Create a contrast transfer function for a complex 3D object")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        param = spider_session.generate_ctf_param(defocus, cs, window, source, defocus_spread, ampcont, envelope_half_width, **extra)
        session.invoke('tf c3', spider_image(outputfile), *param)
        return outputfile
    
    def tf_cor(session, inputfile, ctffile, outputfile=None, **extra):
        '''2D & 3D CTF correction of a series of images/volumes by Wiener 
        filtering. Accumulates a CTF corrected sum over all input images/volumes. 
        Then applies FFT back transform to the accumulated sum. Similar to 
        operation: 'TF CTS' without conjugate multiplication and SNR adjustment.
                
        `Original Spider (TF COR) <http://www.wadsworth.org/spider_doc/spider/docs/man/tfcor.html>`_
        
        .. todo::
            
            Report inconsistency
            Use mu and ft for older spider versions
            
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    File path to input image or volume
        ctffile : str
                  File containing the contrast transfer function
        outputfile : str
                    Filename to store ctf corrected image or volume (If none, temporary incore file is used and returned) (Defaut: None)
        extra : dict
                Unused key word arguments

        :Returns:
        
        outputfile : str
                    CTF corrected image
        '''
        
        _logger.debug("CTF correct 2D or 3D object with back transform")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        session.invoke('tf cor', spider_image(inputfile), spider_image(ctffile), spider_image(outputfile))
        return outputfile
    
    def tf_c(session, defocus, cs, window=None, source=None, defocus_spread=None, ampcont=None, envelope_half_width=None, outputfile=None, **extra):
        '''To compute the phase contrast transfer function for bright-field electron 
        microscopy. The 'TF C' operation produces the transfer function in complex 
        form so that it can be applied (by using 'MU') to the Fourier transform of a model 
        object, for simulations of bright-field weak phase contrast.
                
        `Original Spider (TF C) <http://www.wadsworth.org/spider_doc/spider/docs/man/tfc.html>`_
        
        .. note::
        
            #. Except for `defocus`, all other parameters come directly from the SPIDER params file
            #. Electron wavelength is calculated from the microscope `voltage`
            #. Maximum spatial frequency is calculated from the angstrom per pixel
            #. Used for phase flipping
        
        .. note::
        
            This command is not intended to be used with `spider.setup_options`, 
            use `spider_params.read_parameters_to_dict` to setup the required parameters.
        
        .. seealso:: spider_session.generate_ctf_param
        
            For all required CTF parameters
        
        :Parameters:
        
        session : Session
                  Current spider session
        defocus : float
                  Amount of defocus, in Angstroems
        cs : object
             Spherical aberration constant
        window : int
                 Dimension of the 2D array
        source : float
                 Size of the illumination source in reciprocal Angstroems
        defocus_spread : float
                         Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
        ampcont : float
                  Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
        envelope_half_width : float
                              Envelope parameter specifies the 2 sigma level of the Gaussian
        outputfile : str
                    Filename to store the computed function (If none, temporary incore file is used and returned) (Defaut: None)
        extra : dict
                Unused key word arguments

        :Returns:
        
        outputfile : str
                    Output image of CTF model
        '''
        
        _logger.debug("Create a contrast transfer function for phase flipping with padding")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        param = spider_session.generate_ctf_param(defocus, cs, (window, window), source, defocus_spread, ampcont, envelope_half_width, **extra)
        session.invoke('tf c', spider_image(outputfile), *param)
        return outputfile
    
    def tf_ct(session, defocus, cs, window=None, source=None, defocus_spread=None, ampcont=None, envelope_half_width=None, outputfile=None, **extra):
        '''To compute the phase contrast transfer function for bright-field electron microscopy. The 
        'TF CT' option produces a binary or two-valued (-1,1) transfer function in complex form. 
        This function can be applied (by using 'MU') to the Fourier transform of an object for 
        correcting the phase of bright-field weak phase contrast.
                
        `Original Spider (TF CT) <http://www.wadsworth.org/spider_doc/spider/docs/man/tfct.html>`_
        
        .. note::
        
            #. Except for `defocus`, all other parameters come directly from the SPIDER params file
            #. Electron wavelength is calculated from the microscope `voltage`
            #. Maximum spatial frequency is calculated from the angstrom per pixel
            #. Used for phase flipping
        
        .. note::
        
            This command is not intended to be used with `spider.setup_options`, 
            use `spider_params.read_parameters_to_dict` to setup the required parameters.
        
        .. seealso:: spider_session.generate_ctf_param
        
            For all required CTF parameters
        
        :Parameters:
        
        session : Session
                  Current spider session
        defocus : float
                  Amount of defocus, in Angstroems
        cs : object
             Spherical aberration constant
        window : int
                 Dimension of the 2D array
        source : float
                 Size of the illumination source in reciprocal Angstroems
        defocus_spread : float
                         Estimated magnitude of the defocus variations corresponding to energy spread and lens current fluctuations
        ampcont : float
                  Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
        envelope_half_width : float
                              Envelope parameter specifies the 2 sigma level of the Gaussian
        outputfile : str
                    Filename to store the computed function (If none, temporary incore file is used and returned) (Defaut: None)
        extra : dict
                Unused key word arguments

        :Returns:
        
        outputfile : str
                    Output image of CTF model
        '''
        
        _logger.debug("Create a contrast transfer function for phase flipping with padding")
        if outputfile is None: outputfile = session.temp_incore_image(hook=session.de)
        param = spider_session.generate_ctf_param(defocus, cs, window, source, defocus_spread, ampcont, envelope_half_width, **extra)
        session.invoke('tf ct', spider_image(outputfile), *param)
        return outputfile

    def tf_ed(session, inputfile, apix, cs, ampcont, lam, voltage=None, elambda=None, outputfile=None, **extra):
        '''Estimates the defocus, astigmatism, and cutoff frequency of high frequencies based on 
           2-D power spectrum. Outputs to doc. file and to operation line registers.
                
        `Original Spider (TF ED) <http://www.wadsworth.org/spider_doc/spider/docs/man/tfed.html>`_
        
        .. note::
        
            - This command is not intended to be used with `spider.setup_options`, 
              use `spider_params.read_parameters_to_dict` to setup the required parameters.
            - This command supports incore document files
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        apix : float
               Size of pixel in angstroms
        cs : float
             Spherical aberration constant
        ampcont : float
                  Amplitude constant for envelope parameter specifies the 2 sigma level of the Gaussian
        lam : float
              Wavelength of the electrons
        voltage : float
                  Voltage of microscope
        elambda : float
                  Wavelength of the electrons
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        angle : float
                Angle of astigmatism
        magnitude : float
                    Magnitude of astigmatism
        astdefocus : float
                     Astigmatism corrected defocus
        defocus : float
                  Overall defocus
        cutoff : float
                 Cutoff frequency in 1/A
        outputfile : str
                     Output filename
        '''
        
        _logger.debug("Estimate the transfer function:  defocus, astigmatism, and cutoff frequency of high frequencies: %f"%apix)
        if outputfile is None: 
            width, = session.fi_h(inputfile, ('NSAM'))
            outputfile = session.sd_ic_new((4, width / 2))
        session.invoke('tf ed x21,x22,x23,x24,x25', spider_image(inputfile), spider_tuple(apix, cs), spider_tuple(lam), spider_tuple(ampcont), spider_doc(outputfile))
        return (session['x21'], session['x22'], session['x23'], session['x24'], session['x25'], outputfile)
        
    def ud(session, inputfile, key, nreg=1, out=None, **extra):
        '''Read a set of values from a document file
                
        `Original Spider (UD) <http://www.wadsworth.org/spider_doc/spider/docs/man/ud.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        key : int
              Key for row in document file
        nreg : int
               Number of registers or columns in the document file
        out : ndarray
              Array containing all the entries in the document
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        out : ndarray
              Array containing all the entries in the document
        '''
        
        if out is None: out = numpy.zeros((nreg))
        regs = ["[x%d]"%(i+1) for i in xrange(nreg)]
        session.invoke(('ud %d,'%key)+",".join(regs), spider_doc(inputfile))
        if int(session[9]) > 0: raise spider_session.SpiderCommandError, "ud failed to get value from document file"
        for i in xrange(nreg): out[i] = session[regs[i]]
        return out

    def ud_e(session, inputfile, **extra):
        '''Terminate access to the current document file and allow futher 'UD' operations to access a different document file.
                
        `Original Spider (UD E) <http://www.wadsworth.org/spider_doc/spider/docs/man/ude.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        extra : dict
                Unused key word arguments
        '''
        
        session.invoke('ud e')
    
    def ud_ic(session, inputfile, key, nreg=1, out=None, **extra):
        '''Read a set of values from a document file (load file in core)
                
        `Original Spider (UD IC) <http://www.wadsworth.org/spider_doc/spider/docs/man/udic.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        key : int
              Key for row in document file
        nreg : int
               Number of registers or columns in the document file
        out : ndarray
              Array containing all the entries in the document
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        out : ndarray
              Array containing all the entries in the document
        '''
        
        if out is None: out = numpy.zeros((nreg))
        regs = ["[x%d]"%(i+1) for i in xrange(nreg)]
        session.invoke('ud ic %d,%s'%(key,",".join(regs)), spider_doc(inputfile))
        if int(session[9]) > 0: raise spider_session.SpiderCommandError, "ud ic failed to get value from document file"
        for i in xrange(nreg): out[i] = session[regs[i]]
        return out 

    def ud_ice(session, inputfile, **extra):
        '''Terminate access to the current document file and allow futher 'UD IC' operations to access a different document file.
                
        `Original Spider (UD ICE) <http://www.wadsworth.org/spider_doc/spider/docs/man/udice.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        extra : dict
                Unused key word arguments
        '''
        
        _logger.debug("Delete incore file %s"%str(inputfile))
        if is_incore_filename(inputfile):
            if hasattr(inputfile, 'hook'): inputfile.hook = None
        session.invoke('ud ice', spider_doc(inputfile))
 
    def ud_n(session, inputfile, **extra):
        '''Find highest key, number of columns, and number of keys used in a document file.
        
        `Original Spider (UD N) <http://www.wadsworth.org/spider_doc/spider/docs/man/udn.html>`_
        
        :Parameters:
            
        session : Session
                  Current spider session
        inputfile : str
                    Filename for query file
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        maxkey : int
                 Maximum key in the document file
        ncols : int
                Number of columns in the document file
        nrows : int
                Number of rows in the document file
        '''
        
        if isinstance(inputfile, int): return inputfile, 0, inputfile+1
        if isinstance(inputfile, tuple): return inputfile[1], 0, inputfile[1]-inputfile[0]+1
        session.invoke('ud n [x1],[x2],[x3]', spider_doc(inputfile))
        return int(session['x1']), int(session['x2']), int(session['x3'])
            
    def vo_ea(session, theta_delta=15.0, theta_start=0.0, theta_end=90.0, phi_start=0.0, phi_end=359.999, outputfile=None, **extra):
        '''Create angular document file containing three Eulerian angles defining 
        quasi-evenly spaced projection directions. This document file can be used to 
        create reference projections from a volume (using operation: 'PJ 3Q') for 
        alignment and supervised classification. 
                
        `Original Spider (VO EA) <http://www.wadsworth.org/spider_doc/spider/docs/man/voea.html>`_
        
        .. note::
        
            #. Attempts to determine an error condition based on the register variable, number of angles
            #. Supports incore document files
        
        :Parameters:
        
        session : Session
                  Current spider session
        theta_delta : int
                      Angular step for the theta angles
        theta_start : int
                      Start of theta angle range
        theta_end : int
                    End of theta angle range
        phi_start : int
                    Start of phi angle range
        phi_end : int
                  End of phi angle range
        outputfile : str
                     Filename for the angle output (If none, temporary incore file is used and returned)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     File path to an output file
        num_angles : int
                     Number of angles generated
        '''
        
        _logger.debug("Generating evenly space angles")
        if outputfile is None: 
            assert(False)
            outputfile = session.sd_ic_new( (3, session.vo_ea_n(theta_delta, theta_start, theta_end, phi_start, phi_end)) )
        else:
            try: session.de(outputfile)
            except: pass
        session.invoke('vo ea x11', spider_tuple(theta_delta), spider_tuple(theta_start, theta_end), spider_tuple(phi_start, phi_end), spider_doc(outputfile))
        if int(session['x11']) < 1: raise spider_session.SpiderCommandError, "No angles produced"
        if int(session[9]) > 0: raise spider_session.SpiderCommandError, "vo ea failed to produce an output file"
        return (outputfile, int(session['x11']))
    
    def vo_ras(session, inputfile, angle_num, rotation, psi_value=(1,0), outputfile=None, **extra):
        '''Rotate projection directions according to three rotation angles 
        supplied. The original projection directions are provided in the 
        input angular document file. The modified projection directions 
        are stored in the output angular document file. If this output 
        angular docfile is used in a subsequent 3D reconstruction, the 
        resulting structure will be rotated in 3D space in agreement with 
        the three rotation angles given. This command is provided to calculate 
        "merged" 3D reconstructions and can force a particular output angle to 
        be set to a particular value.
                
        `Original Spider (VO RAS) <http://www.wadsworth.org/spider_doc/spider/docs/man/voras.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Name of the input angle file
        angle_num : int
                    Number of angles
        rotation : (float,float,float)
                   Rotation in terms of (PHI,THETA,PSI)
        psi_value : (int,float)
                    Column to set to zero
        outputfile : str
                     Filename for the angle output (If none, temporary incore file is used and returned)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     File path to an output file
        '''
        
        _logger.debug("Generating rotated angles")
        if outputfile is None: 
            outputfile = session.sd_ic_new( (3, angle_num) )
        else:
            try: session.de(outputfile)
            except: pass
        session.invoke('vo ras', spider_doc(inputfile), spider_tuple(*angle_num), spider_tuple(*psi_value), spider_doc(outputfile))
        return outputfile
    
    def wi(session, inputfile, dimensions, coords=None, outputfile=None, **extra):
        '''Cut out a window from a specified image/volume file.
                
        `Original Spider (WI) <http://www.wadsworth.org/spider_doc/spider/docs/man/wi.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input file template, stack template or stack
        dimensions : (int,int,int)
                     Width, height and depth of the slice (NSAM, NROW, NSLICE)
        coords : (int,int,int)
                 X, Y, Z coordinate for top left origin of the window
        outputfile : str
                     Filename for incore stack (Default: None)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
                
        #stack_count, z_size, x_size, y_size, = session.fi_h(spider_stack(inputfile), ('MAXIM', 'NSLICE', 'NSAM', 'NROW'))
        z_size, x_size, y_size, = session.fi_h(inputfile, ('NSLICE', 'NSAM', 'NROW'))
        #if stack_count > 1: raise ValueError, "Does not support stacks"
        
        if z_size > 1:
            # TODO - add dimension_tuple
            if not hasattr(dimensions, '__len__'): dimensions = (dimensions, dimensions, dimensions)
            elif len(dimensions) == 1: dimensions = (dimensions[0], dimensions[0], dimensions[0])
            elif len(dimensions) == 2: dimensions = (dimensions[0], dimensions[1], dimensions[0])
        else:
            if not hasattr(dimensions, '__len__'): dimensions = (dimensions, dimensions, 1)
            elif len(dimensions) == 1: dimensions = (dimensions[0], dimensions[0], 1)
            elif len(dimensions) == 2: dimensions = (dimensions[0], dimensions[1], 1)
        
        if coords is None:
            if z_size > 1:
                coords = (int(x_size-dimensions[0])/2, int(y_size-dimensions[1])/2, int(z_size-dimensions[0])/2)
            else:
                coords = (int(x_size-dimensions[0])/2, int(y_size-dimensions[1])/2)
        else:
            if z_size > 1:
                if not hasattr(coords, '__len__'): coords = (coords, coords, coords)
                elif len(coords) == 1: coords = (coords[0], coords[0], coords[0])
                elif len(coords) == 2: coords = (coords[0], coords[1], coords[0])
            else:
                if isinstance(coords, int): coords = (coords, coords, 1)
                elif len(coords) == 1: coords = (coords[0], coords[0], 1)
                elif len(coords) == 2: coords = (coords[0], coords[1], 1)
        return spider_session.spider_command_fifo(session, 'wi', inputfile, outputfile, "Window out subimage", spider_coord_tuple(dimensions), spider_coord_tuple(coords))
    
    def wu(session, inputfile, outputfile=None, **extra):
        '''Takes the square root of a image.
        
        alias: square_root
        
        `Original Spider (WU) <http://www.wadsworth.org/spider_doc/spider/docs/man/wu.html>`_
        
        :Parameters:
        
        session : Session
                  Current spider session
        inputfile : str
                    Input filename
        outputfile : str
                     Output filename (If not specified (None), then it creates an incore-file)
        extra : dict
                Unused key word arguments
        
        :Returns:
        
        outputfile : str
                     Output filename
        '''
        
        return spider_session.spider_command_fifo(session, 'wu', inputfile, outputfile, "Take square root of an image")

def supports_internal_rtsq(session):
    ''' TEst if the SPIDER version supports internal RTSQ
    
    :Parameters:
        
    session : Session
              Current spider session
    
    :Returns:
    
    support : bool
              True if the version support internal RTSQ
    '''
    
    return session.version[0] >= 19 and session.version >= 11 and 1 ==0

def max_translation_range(window, ring_last, **extra):
    ''' Ensure a valid translation range
    
    :Pararameters:
    
    window : int
             Size of the window
    ring_last : int
                Last alignment ring
    
    :Returns:
    
    trans_range : int
                  Max translation range
    '''
    
    return int(window/2.0) - ring_last - 3

def ensure_proper_parameters(param):
    ''' Ensure certain SPIDER parameters have the proper values
    
    :Parameters:
    
    param : dict
            Parameter dictionary
    '''
    
    if 'ring_last' in param:
        if param['ring_last'] == 0 or param['ring_last'] > (param['window']/2.0): 
            param['ring_last'] = int(param['pixel_diameter']/2.0)
        
        if (param['window']/2 - param['ring_last'] - param['trans_range']) < 3:
            param['trans_range'] = max_translation_range(**param)
            if (param['trans_range']%param['trans_step']) > 0:
                val = param['trans_range']-(param['trans_range']%param['trans_step'])
                if val > 3: param['trans_range']=val
                else: param['trans_step']=1
      
def count_images(session, filename):
    ''' Count number of images in a SPIDER file
    
    :Parameters:
        
    session : Session
              Current spider session
    filename : str
               Name of stack to count
    
    :Returns:
    
    stack_count : int
                  Number of images in the stack
    '''
    
    stack_count,  = session.fi_h(spider_stack(filename), ('MAXIM'))
    return stack_count

def image_size(session, filename):
    ''' Count number of images in a SPIDER file
    
    :Parameters:
        
    session : Session
              Current spider session
    filename : str
               Name of input file
    
    :Returns:
    
    shape : tuple
            Shape of the image
    '''
    
    ''' Add code to detect SPIDER stack?
    istack, = session.fi_h(filename, ('ISTACK', ))
    if istack > 0:
        shape  = session.fi_h(spider_stack(filename), ('NSAM','NROW', 'NSLICE'))
    else:
    '''
    shape  = session.fi_h(spider_image(filename), ('NSAM','NROW', 'NSLICE'))
    return shape

def for_image(session, filename, count=None):
    ''' Iterate through a SPIDER stack, yeild an incore image
    
    :Parameters:
        
    session : Session
              Current spider session
    filename : str
               Name of stack to count
    
    :Returns:
    
    image : spider_var
            Incore SPIDER image
    '''
    
    if count is None: count = count_images(session, filename)
    image = None
    for i in xrange(count):
        image = session.cp((filename, i+1), outputfile=image)
        yield image
        
def nonspi_file(session, filename, temp):
    ''' Add extension to filename, in addition write to temp file if file is incore
    
    :Parameters:
        
    session : Session
              Current spider session
    filename : str
               Name of filename to convert
    temp : str
           Temporary file to contain incore file
    
    :Returns:
    
    image : spider_var
            Incore SPIDER image
    '''
    
    if is_incore_filename(filename):
        session.cp(filename, temp)
        filename=temp
    return session.replace_ext(filename)

def open_session(args, spider_path="", data_ext="", thread_count=0, enable_results=False, rank=None, local_temp=None, **extra):
    '''Opens a spider session
    
    :Parameters:
        
        args : list
               List of input file names
        spider_path : str
                      File path to spider executable
        data_ext : str
                   Extension of spider data files
        thread_count : int, noupdate
                       Number of threads per machine, 0 means use all cores
        enable_results : bool, noupdate
                         If set true, print results file to terminal
        rank : int
               MPI node rank
        local_temp : str
                     Path to start SPIDER
        extra : dict
                Unused keyword arguments
    
    :Returns:
        
        session : Session
                  A spider sesssion with all available commands
    '''
    
    if args is not None and len(args) > 0:
        tmp_ext = os.path.splitext(args[0])[1][1:]
        if data_ext != tmp_ext and data_ext == "":
            if rank is None or rank == 0:
                _logger.warn("Changing Spider data extension from %s to %s"%(data_ext, tmp_ext))
            data_ext = tmp_ext
        elif data_ext == "": data_ext = 'spi'
    return Session(spider_path, data_ext, thread_count, enable_results, rank, local_temp)

def stack(session, inputfile, node_count, outputfile=None):
    ''' Stack a set of individual spider stacks into a single stack
    
    This code assumes the numbering starts a 1 and goes to `node_count`.
    
    :Parameters:
        
    session : Session
              Current spider session
    inputfile : list or dict
                List of input files in pySPIDER template format, file_0000
    node_count : int
                 Number of spider stacks
    outputfile : str
                 Output file path
    
    :Returns:
    
    outputfile : str
                 Output file path
    stack_total : int
                  Number of projections in output stack
    '''
    
    if node_count < 2: 
        curr_count, = session.fi_h(spider_stack(inputfile[0]), ('MAXIM'))
        return inputfile[0], curr_count
    stack_count = []
    stack_total = 0
    for i in xrange(1, node_count):
        curr_count, = session.fi_h(spider_stack(inputfile[i]), ('MAXIM'))
        stack_count.append(int(curr_count))
        stack_total += int(curr_count)
    
    if outputfile is None: outputfile = session.ms(stack_total, spider_stack( (inputfile[0], 1) ))
    start = 1 if inputfile == outputfile else 0
    stack_offset = 0
    for i in xrange(start, node_count):
        for j in xrange(stack_count[i]):
            session.cp((inputfile[i], j+1), outputfile=(outputfile, stack_offset))
            stack_offset += 1
    return outputfile, stack_total

def enumerate_stack(inputfile, selection, outputfile=None):
    '''Loop over a spider stack, creating the proper slice name. It counts
    the number of slices in the stack.
    
    .. sourcecode:: py
        
        for input in enumerate_stack(inputfile):
            ft_stack_slice = spi.ft(input)
    
    :Parameters:
        
    inputfile : str (or dict)
                Input filename to iterate over
    selection : int or array
                Size of stack or array of selected indices (if inputfile dict, then must be 2D)
    outputfile : str, optional
                 Output filename to include
    
    :Returns:
    
    inputslice : generator
                 Tuple filename and index
    outputslice : generator
                  Tuple filename and index (If outputfile is not None)
    '''
    
    if not hasattr(selection, '__iter__'): selection = xrange(1, selection+1)
    if isinstance(inputfile, dict):
        if outputfile is None:
            for fid, index in selection: 
                yield (inputfile[int(fid)], int(index)) 
        else:
            for j, index in enumerate(selection): 
                fid, index = index
                yield (inputfile[int(fid)], int(index)), (outputfile, int(j+1))
    else:
        if outputfile is None:
            for index in selection: yield (inputfile, int(index)) 
        else:
            for j, index in enumerate(selection): yield (inputfile, int(index)), (outputfile, int(j)+1)

def phase_flip(session, inputfile, defocusvals, outputfile, mult_ctf=False, rank=0, **extra):
    ''' Phase flip the input stack and store in the given output stack
    
    :Parameters:
    
    session : Session
              Current spider session
    inputfile : str (or dict)
                Input filename to iterate over
    defocusvals : array
                 Array of defocus values that correspond to the input stack
    outputfile : str, optional
                 Output filename to include
    mult_ctf : bool
              Multiply by the CTF rather than phase flip
    rank : int
            Current process rank
    extra : dict
            Unused keyword arguments
    '''
    
    defocus = 0
    ftimage = None
    ctfimage = None
    ctf = None
    if rank == 0: _logger.info("Generating phase flip stack")
    session.de(outputfile)
    
    stack_count, = session.fi_h(spider_stack(inputfile), ('MAXIM', ))
    for inputfile, outputfile in enumerate_stack(inputfile, int(stack_count), outputfile):
        i = inputfile[1]-1
        if defocus != float(defocusvals[i]):
            defocus = float(defocusvals[i])
            if mult_ctf:
                ctf = session.tf_c(defocus=defocus, outputfile=ctf, **extra)
            else:
                ctf = session.tf_ct(defocus=defocus, outputfile=ctf, **extra)
        ftimage = session.ft(inputfile, outputfile=ftimage, **extra)           # Fourier transform reference volume
        ctfimage = session.mu(ftimage, ctf, outputfile=ctfimage)               # Multiply volume by the CTF
        session.ft(ctfimage, outputfile=outputfile) 

def scale_parameters(bin_factor, dec_level=1.0, pj_radius=-1, trans_range=24, trans_step=1, first_ring=1, ring_last=0, ring_step=1, cg_radius=0, window=0, **extra):
    ''' Scale parameters that depend on the window size
    
    :Parameters:
    
    bin_factor : float
                 Current decimation factor
    dec_level : int
                Previous decimation level
    pj_radius : int
                Radius of sphere to compute projection, if less than one use 0.69 times the diameter of the object in pixels (Default: -1)
    trans_range : float
                  Maximum allowed translation; if this value exceeds the window size, then it will lowered to the maximum possible
    trans_step : float
                 Translation step size
    first_ring : int
                 First polar ring to analyze
    ring_last : int
                Last polar ring to analyze; if this value is zero, then it is chosen to be the radius of the particle in pixels
    ring_step : int
                Polar ring step size
    cg_radius : int
                Radius of reconstructed object
    window : int
             Current window size
    extra : dict
            Unused keyword arguments
            
    :Returns:
    
    param : dict
            Dictionary of updated parameters
    '''
    
    if dec_level == bin_factor: return {}
    max_radius = int(window/2.0)
    param = {}
    factor = dec_level/bin_factor
    if pj_radius > 0: param['pj_radius']=min(int(pj_radius*factor), max_radius)
    if trans_range > 1: param['trans_range']=max(1, int(trans_range*factor))
    if trans_step > 1: param['trans_step']=max(1, int(trans_step*factor))
    if first_ring > 1: param['first_ring']=max(1, int(first_ring*factor))
    if ring_last > 0:
        param['ring_last']=min(max_radius - 4, int(ring_last*factor))
    if ring_step > 1: param['ring_step']=max(1, int(ring_step*factor))
    if cg_radius > 0: param['cg_radius']=min(int(cg_radius*factor), max_radius)
    if (max_radius - param['ring_last'] - param['trans_range']) < 3:
        if param['trans_range'] > 1:
            param['trans_range'] = max(1, max_radius - param['ring_last'] - 3)
        if (max_radius - param['ring_last'] - param['trans_range']) < 3:
            param['ring_last'] = max(2, max_radius - param['trans_range'] - 3)
        
    param['dec_level']=bin_factor
    return param

def cache_data(session, inputfile, selection, outputfile, window, rank=0):
    ''' Test if output file is the correct size and has the correct number of images, if not then
    copy the images in the correct size and number
    
    .. todo:: write selection file and compare when not writing
    
    :Parameters:
    
    session : Session
              Current spider session
    inputfile : str (or dict)
                Input filename to iterate over
    selection : int or array
                Size of stack or array of selected indices (if inputfile dict, then must be 2D)
    outputfile : str, optional
                 Output filename to include
    window : int
             New size each window
    rank : int
            Current process rank
    
    :Returns:
    
    copied : bool
             True if a copy was performed
    '''
    
    window = int(window)
    if not is_incore_filename(outputfile) and os.path.exists(session.replace_ext(outputfile)):
        width, stack_count = session.fi_h(spider_stack(outputfile), ('NSAM', 'MAXIM'))
        stack_count = int(stack_count)
        width = int(width)
    else: stack_count = 0
    selection_file = os.path.splitext(outputfile)[0]+'.csv'
    selection_file = os.path.join(os.path.dirname(selection_file), "sel_"+os.path.basename(selection_file))
    
    remote_select = numpy.loadtxt(selection_file, delimiter=",") if os.path.exists(selection_file) else None
    if stack_count != len(selection) or stack_count != remote_select.shape[0] or width != window or remote_select is None or not numpy.alltrue(remote_select == selection):
        if rank == 0:
            if remote_select is None:  _logger.info("Transfering data to cache - selection file not found")
            elif width != window:  _logger.info("Transfering data to cache - incorrect window size - %d != %d"%(width, window))
            elif stack_count != len(selection): _logger.info("Transfering data to cache - incorrect number of windows - %d != %d"%(stack_count, len(selection)))
            else: _logger.info("Transfering data to cache - selection files do not match")
        input = inputfile if not isinstance(inputfile, dict) else inputfile[inputfile.keys()[0]]
        width = session.fi_h(spider_stack(input), ('NSAM', ))
        copy(session, inputfile, selection, outputfile)
        stack_count,  = session.fi_h(spider_stack(outputfile), ('MAXIM', ))
        if stack_count == 0: 
            import socket
            raise ValueError, "No images copied for %s"%socket.gethostname()
        numpy.savetxt(selection_file, selection, delimiter=",")
        return True
    return False

def cache_interpolate(session, inputfile, selection, outputfile, window, rank=0):
    ''' Test if output file is the correct size and has the correct number of images, if not then
    copy the images in the correct size and number
    
    .. todo:: write selection file and compare when not writing
    
    :Parameters:
    
    session : Session
              Current spider session
    inputfile : str (or dict)
                Input filename to iterate over
    selection : int or array
                Size of stack or array of selected indices (if inputfile dict, then must be 2D)
    outputfile : str, optional
                 Output filename to include
    window : int
             New size each window
    rank : int
            Current process rank
    
    :Returns:
    
    copied : bool
             True if a copy was performed
    '''
    
    window = int(window)
    if not is_incore_filename(outputfile) and os.path.exists(session.replace_ext(outputfile)):
        width, stack_count = session.fi_h(spider_stack(outputfile), ('NSAM', 'MAXIM'))
        stack_count = int(stack_count)
        width = int(width)
    else: stack_count = 0
    selection_file = os.path.splitext(outputfile)[0]+'.csv'
    selection_file = os.path.join(os.path.dirname(selection_file), "sel_"+os.path.basename(selection_file))
    
    remote_select = numpy.loadtxt(selection_file, delimiter=",") if os.path.exists(selection_file) else None
    if stack_count != len(selection) or stack_count != remote_select.shape[0] or width != window or remote_select is None or not numpy.alltrue(remote_select == selection):
        if rank == 0:
            if remote_select is None:  _logger.info("Transfering data to cache - selection file not found")
            elif width != window:  _logger.info("Transfering data to cache - incorrect window size - %d != %d"%(width, window))
            elif stack_count != len(selection): _logger.info("Transfering data to cache - incorrect number of windows - %d != %d"%(stack_count, len(selection)))
            else: _logger.info("Transfering data to cache - selection files do not match")
        input = inputfile if not isinstance(inputfile, dict) else inputfile[inputfile.keys()[0]]
        width = session.fi_h(spider_stack(input), ('NSAM', ))
        if width != window:
            copy_interpolate(session, inputfile, selection, outputfile, window)
        else:
            copy(session, inputfile, selection, outputfile)
        stack_count,  = session.fi_h(spider_stack(outputfile), ('MAXIM', ))
        if stack_count == 0: 
            import socket
            raise ValueError, "No images copied for %s"%socket.gethostname()
        numpy.savetxt(selection_file, selection, delimiter=",")
        return True
    return False

def interpolate_stack(session, inputfile, outputfile=None, window=0, **extra):
    ''' Safely copy the input image to match the proper window size
    
    :Parameters:
    
    session : Session
              Current spider session
    inputfile : str (or dict)
                Input filename for image
    outputfile : str, optional
                 Output filename
    window : int
             New size each window
    
    :Returns:
    
    outputfile : str
                 Filename of output image
    '''
    
    if outputfile is not None and os.path.exists(session.replace_ext(outputfile)):
        width, count = session.fi_h(spider_stack(inputfile), ('NSAM', 'MAXIM'))
        width, count = int(width), int(count)
        if window == width and count == int(session.fi_h(spider_stack(session.replace_ext(outputfile)), ('MAXIM', ))[0]): return outputfile
    width, count = session.fi_h(spider_stack(inputfile), ('NSAM', 'MAXIM'))
    width, count = int(width), int(count)
    if window != width:
        session.de(outputfile)
        for i in xrange(1, count+1):
            session.ip(spider_image(inputfile, i), (window, window), outputfile=spider_image(outputfile, i))
        return outputfile
    return inputfile

def copy_safe(session, inputfile, window=0, **extra):
    ''' Safely copy the input image to match the proper window size
    
    :Parameters:
    
    session : Session
              Current spider session
    inputfile : str (or dict)
                Input filename for image
    window : int
             New size each window
    
    :Returns:
    
    outputfile : str
                 Filename of output image
    '''
        
    width, = session.fi_h(inputfile, ('NSAM'))
    if window != width:
        return session.ip(inputfile, (window, window, window), **extra)
    else:
        return session.cp(inputfile, **extra)

def copy_interpolate(session, inputfile, selection, outputfile, window):
    ''' Copy an interpolated set of stacks to an output stack
    
    :Parameters:
    
    session : Session
              Current spider session 
    inputfile : str (or dict)
                Input filename to iterate over
    selection : int or array
                Size of stack or array of selected indices (if inputfile dict, then must be 2D)
    outputfile : str, optional
                 Output filename to include
    window : int
             New size each window
    '''
    
    session.de(outputfile)
    tempout = None
    for inputfile, outputfile in enumerate_stack(inputfile, selection, outputfile):
        tempout=session.ip(inputfile, (window, window), outputfile=tempout)
        session.cp(tempout, outputfile=outputfile)

def copy(session, inputfile, selection, outputfile):
    ''' Copy a set of stacks to an output stack
    
    :Parameters:
    
    session : Session
              Current spider session  
    inputfile : str (or dict)
                Input filename to iterate over
    selection : int or array
                Size of stack or array of selected indices (if inputfile dict, then must be 2D)
    outputfile : str, optional
                 Output filename to include
    '''
    
    session.de(outputfile)
    for inputfile, outputfile in enumerate_stack(inputfile, selection, outputfile):
        session.cp(inputfile, outputfile=outputfile)

def angle_count(theta_delta=15.0, theta_start=0.0, theta_end=90.0, phi_start=0.0, phi_end=359.9, **extra):
    '''Count number of angles returned by vo ea.
    
    :Parameters:
        
        theta_delta : int
                      Angular step for the theta angles
        theta_start : int
                      Start of theta angle range
        theta_end : int
                    End of theta angle range
        phi_start : int
                    Start of phi angle range
        phi_end : int
                  End of phi angle range
        extra : dict
                Unused key word arguments
    
    :Returns:
        
        count : int
                Number of angles
    '''
    
    count = 0
    phi_range = phi_end-phi_start
    skip = (theta_start < 90.0) and (theta_end == 90.0) and (phi_start == 0.0) and (phi_end > 180.0)
    for theta in numpy.arange(theta_start, theta_end+1, theta_delta):
        if theta == 0.0 or theta == 180.0:
            phi_delta = 360.0
            phi_count = 1
        else:
            phi_delta = theta_delta / numpy.sin( numpy.deg2rad(theta) )
            phi_count = max( int(phi_range/phi_delta)-1, 1 )
            phi_delta = phi_range / float(phi_count)
        for i in xrange(1, phi_count+1):
            phi = phi_start+(i-1) * phi_delta
            if skip and theta == 90.0 and phi >= 180.0: continue
            count += 1
    return count

def throttle_mp(spi, thread_count, **extra):
    ''' Set number of cores in SPIDER to 1
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    thread_count : int
                   Number of threads to use
    extra : dict
            Unused keyword arguments 
    '''
    
    if thread_count > 1 or thread_count == 0: spi.md('SET MP', 1)

def release_mp(spi, thread_count, **extra):
    ''' Reset number of cores in SPIDER to default
    
    :Parameters:
    
    spi : spider.Session
          Current SPIDER session
    thread_count : int
                   Number of threads to use
    extra : dict
            Unused keyword arguments 
    '''
    
    if thread_count > 1 or thread_count == 0: spi.md('SET MP', thread_count)



