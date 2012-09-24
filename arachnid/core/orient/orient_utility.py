''' Utilities to convert between orientation parameters

.. Created on Sep 24, 2012
.. codeauthor:: robertlanglois
'''
import numpy

def align_2D_to_3D(rot, tx, ty):
    ''' Convert 2D to 3D alignment parameters
    
    TR -> RT
    
    :Parameters:
    
    rot : float
          In plane rotation (TR)
    tx : float
         Translation in the x-direction (TR)
    ty : float
         Translation in the y-direction (TR)
         
    :Returns:
    
    psi : float
          PSI angle (RT)
    sx : float
         Translation in the x-direction (RT)
    sy : float
         Translation in the y-direction (RT)
    '''
    
    ca = numpy.cos(rot)
    sa = numpy.sin(rot)
    sx = tx*ca + ty*sa
    sy = ty*ca - tx*sa
    return -rot, sx, sy