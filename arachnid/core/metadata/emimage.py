'''
.. Created on Jul 21, 2013
.. codeauthor:: robertlanglois
'''
class ctf_emx(object):
    ''' Contrast transfer function (CTF) describing the estimated blur of the image data 
    
    TODO python units code
    '''
    
    def __init__(self):
        '''
        '''
        
        self.amplitudeContrast = 0.0 
        '''Units: N/A
        Type: float
        Description: Describes the amplitude contrast contribution to the image as a fraction. Range 0-1
        '''
        
        self.cs = 0.0
        '''Units: mm
        Type: float
        Description: Spherical aberration of the objective lens of the instrument on which the data was collected. Normally this is a manufacturer-provided value.
        '''
        
        self.defocusU = 0.0
        '''Units: nm
        Type: float
        Description: For underfocus micrographs, this is the defocus along the direction defined by the major axis of the Thon rings. Therefore this is the value of the minimal defocus. Positive numbers indicate underfocus images.
        '''
        
        self.defocusV = 0.0
        '''Units: nm
        Type: float
        Description: For underfocus micrographs, this is the defocus along the direction defined by the minor axis of the Thon rings. Therefore this is the value of the maximal defocus. Positive numbers indicate underfocus images.
        '''
        
        self.defocusUAngle = 0.0
        '''Units: degrees
        Type: float
        Description: For underfocus micrographs, this is the defocus along the direction defined by the minor axis of the Thon rings. Therefore this is the value of the maximal defocus. Positive numbers indicate underfocus images.
        '''
        
        self.pixelSpacing = 0.0
        '''Units: pixel/Ångstroms / px/A
        Type: float
        Description: Sampling rate
        '''
        
        self.voltage = 0.0
        '''Units: kiloVolts / kV
        Type: float
        Description: Describes the acceleration voltage used to take the image
        '''
        
        self.micrograph = ""
        '''Units: N/A
        Type: str
        Description: Micrograph identifier
        '''





class ndimage_emx(object):
    '''
    '''
    
    def __init__(self):
        '''
        '''
        
        self.image_ctf = ctf_emx()
        