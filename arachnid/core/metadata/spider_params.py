''' Parse a SPIDER Parameter file and add the keys to a dictionary

A spider parameter file contains information regarding a Cyro-EM data
collection including CTF and other features.

Parameters
----------

.. option:: -p <FILENAME>, --param-file <FILENAME> 
    
    Filename for SPIDER parameter file describing a Cryo-EM experiment

.. option:: --bin-factor <FLOAT>
    
    Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified

Examples
---------

.. sourcecode:: sh
    
    $ spi-align -p params.dat
    $ more params.dat
     ;spi/dat   11-AUG-2009 AT 16:50:56   params.dat
     ;  KEY:      PARAMETERS FOR SINGLE PARTICLE RECONSTRUCTION                                                                                                    
     ; 1) ZIP FLAG (0 = DO NOT UNZIP, 1 = NEEDS TO BE UNZIPPED)                                                                                                    
        1 1   0.0000     
     ; 2) FILE FORMAT (0:SPIDER, 1:HISCAN TIF, 2:PERKINELMER, 3:ZI SCANNER)                                                                                        
        2 1   0.0000     
     ; 3) MICROGRAPH WIDTH, IN PIXELS (IGNORED IF THIS INFO IS IN THE HEADER)                                                                                      
        3 1   0.0000     
     ; 4) MICROGRAPH HEIGHT, IN PIXELS (IGNORED IF THIS INFO IS IN THE HEADER)                                                                                     
        4 1   0.0000                                                                                                                                                  
    --More--

.. todo:: See if it makes sense to remove read_parameters_to_dict

.. Created on Nov 14, 2010
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import logging, math, numpy
import format

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def write(output, apix, voltage, cs, xmag, pixel_diameter, window_size=0, ampcont=0.1, res=7, envelope_half_width=10000.00, **extra):
    ''' Create a SPIDER params file from a set of parameters
    
    :Parameters:
    
    output : str
             Output filename for the SPIDER params file
    apix : float
           Pixel size, A
    voltage : float
              Electron energy, KeV
    cs : float
         Spherical aberration, mm
    xmag : float
           Magnification
    pixel_diameter : int
                     Actual size of particle, pixels
    window_size : int
                  Particle window size, pixels 
    ampcont : float
              Amplitude contrast ratio
    res : float
          Scanning resolution (7 or 14 microns)
    envelope_half_width : float
                          Gaussian envelope halfwidth, 1/A
    extra : dict
            Unused keyword arguments
    '''
    
    if window_size == 0:
        window_size = pixel_diameter * 1.3
        if (window_size%2)==0: window_size += 1
            
    vals = numpy.zeros(20)
    # 1 - Zip flag (0 = Do not unzip, 1 = Needs to be unzipped)
    # 2 - File format (0:SPIDER, 1:HiScan tif, 2:PerkinElmer, 3:ZI Scanner)
    # 3 - Micrograph width, in pixels (ignored if present in header)
    # 4 - Micrograph height, pixels (ignored if present in header)
    # 5 - Pixel size, A
    vals[4] = apix
    # 6 - Electron energy, KeV
    vals[5] = voltage
    # 7 - Spherical aberration, mm
    vals[6] = cs
    # 8 - Source size, 1/A
    # 9 - Defocus spread, A
    #10 - Astigmatism, A
    #11 - Azimuth of astigmatism, degrees
    #12 - Amplitude contrast ratio
    vals[11] = ampcont
    #13 - Gaussian envelope halfwidth, 1/A
    vals[12] = envelope_half_width
    #14 - Lambda, A
    vals[13] = 12.398 / math.sqrt(voltage * (1022.0 + voltage))
    #15 - Maximum spatial frequency, 1/A
    vals[14] = 0.5 / apix
    #16 - Decimation factor
    vals[15] = 1
    #17 - Particle window size, pixels 
    vals[16] = window_size
    #18 - Actual size of particle, pixels
    vals[17] = pixel_diameter
    #19 - Magnification
    vals[18] = xmag
    #20 - Scanning resolution (7 or 14 microns)
    vals[19] = res
    format.write(output, vals, header=['param'], format=format.spiderdoc)

def read(filename, extra=None):
    '''Read spider parameters
    
    :Parameters:

    filename : str
              File path to parameter file

    extra : dict
            Spider parameter dictionary
    
    :Returns:
    
    val : dict
          Spider parameter dictionary
    '''
    
    param = {}
    if 'comm' not in param or param['comm'] is None or param['comm'].Get_rank() == 0:
        bin_factor = extra.get('bin_factor', 1.0) if extra is not None else 1.0
        #      1    2     3      4      5    6     7    8          9            10        11      12             13         14    15    16   17         18         19  20
        keys="zip,format,width,height,apix,voltage,cs,source,defocus_spread,astigmatism,azimuth,ampcont,envelope_half_width,lam,maxfreq,dec_level,window,pixel_diameter,xmag,res".split(',')
        fin = file(filename, 'r')
        index = 0
        for line in fin:
            line = line.strip()
            if line == "" or line[0] == ';': continue
            param[keys[index]] = float(line.split()[2])
            index += 1
        fin.close()
        _logger.debug("Decimation: %d, %d"%(bin_factor, param['dec_level']))
        if bin_factor > 1.0 and bin_factor != param['dec_level']:
            param.update(update_params(bin_factor, **param))
        _logger.debug("apix: %f"%(param['apix']))
    if 'comm' in param and param['comm'] is not None:
        param = param['comm'].bcast(param)
    if extra is not None: extra.update(param)
    return param

def update_params(bin_factor, width, height, apix, maxfreq, window, pixel_diameter, dec_level, **extra):
    ''' Update the SPIDER params based on the current decimation factor
    
    :Parameters:
    
    bin_factor : float
                 Current decimation factor
    width : int
            Width of the micrograph
    height : int
             Height of the micrograph
    apix : float
           Pixel size
    maxfreq : float
              Maximum spatial frequence
    window : int
             Window size
    pixel_diameter : int
                     Diameter of the particle in pixels
    dec_level : int
                Current decimation level
    extra : dict
            Unused extra keyword arguments
    '''
    
    if dec_level == bin_factor: return {}
    return dict(dec_level=bin_factor,
                width=int(width/bin_factor), 
                height=int(height/bin_factor), 
                apix=apix*bin_factor, 
                maxfreq=maxfreq/bin_factor, 
                window=int(window/bin_factor), 
                pixel_diameter=int(pixel_diameter/bin_factor))
    
    

def ctf_spider2EMAN(apix, ampcont, voltage, window, cs, bfactor=0.0, defocus=0.0, **extra):
    ''' Convert the Spider CTF to EMAN2
    
    :Parameters:

    apix : float
           Pixels per angstrom
    ampcont : float
              Amplitude constant
    voltage : float
              Microscope voltage
    window : int
             Window size in pixels
    cs : float
         Spherical abberation
    bfactor : float
              Structure factor
    defocus : float
              Defocus coefficient
    extra : dict
            Unused extra keyword arguments
    
    :Returns:
    
    val : dict
          CTF parameter dictionary
    '''
    
    #ds=1.0/(apix*window)
    if defocus > 100: defocus *= 1e-4
    if ampcont < 1.0: ampcont = ampcont*100/math.sqrt(2*ampcont**2-2*ampcont+1)
    return {"defocus": defocus, "cs": cs, "voltage": voltage, "apix": apix, "bfactor": bfactor, "ampcont": ampcont}#, "dsbg": ds}

def setup_options(parser, pgroup=None, required=False):
    '''Add options to the given option parser
    
    :Parameters:

    parser : optparse.OptionParser
             Program option parser
    pgroup : optparse.OptionGroup
            Parent option group
    '''
    
    from ..app.settings import OptionGroup
    group = OptionGroup(parser, "Params", "Options to control basic SPIDER parameters", id=__name__, gui=dict(root=True, stacked="prepComboBox"))
    group.add_option("-p", param_file="",  help="Spider parameter file describing a Cryo-EM experiment", required_file=required, gui=dict(filetype="open"))
    group.add_option("-b", bin_factor=1.0, help="Decimatation factor for the script: changes size of images, coordinates, parameters such as pixel_size or window unless otherwise specified")
    if pgroup is not None:
        pgroup.add_option_group(group)
    else:
        parser.add_option_group(group)
    
def update_options(options):
    ''' Update the current options
    
    :Parameters:

    options : object
              Object whose fields are options that hold the corresponding values
    
    '''
    from ..app.settings import OptionValueError
    
    if options.bin_factor == 0.0: raise OptionValueError, "Bin factor cannot be zero (--bin-factor)"
    if options.param_file != "":
        params = read(options.param_file, vars(options))
        if hasattr(options, "pixel_radius") and options.pixel_radius == 0: 
            options.pixel_radius = int(params["pixel_diameter"]/2.0)
        for key, val in params.iteritems():
            setattr(options, key, val)

def check_options(options, main_option=False):
    '''Check if the option values are valid
    
    This function tests if the option values are valid and performs any
    pre-processing of the options before the program is run.
    
    :Parameters:

    options : object
              Object whose fields are options that hold the corresponding values
    main_option : bool
                  Ignored
    '''
    from ..app.settings import OptionValueError
    
    if options.bin_factor == 0.0: raise OptionValueError, "Bin factor cannot be zero (--bin-factor)"

