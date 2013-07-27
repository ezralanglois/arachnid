''' Definition of the ndimage objects

I2PC/NCMI EMX standard

.. Created on Aug 14, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import numpy

class ndimage(numpy.ndarray):
    ''' Contains header information for an image
    '''
    
    __slots__=('header')
    
    def __new__(cls, input_array, header=None):
        '''
        '''
        
        obj = numpy.asarray(input_array).view(cls)
        obj.header=header
        return obj
    
    def __array_finalize__(self, obj):
        '''
        '''
        
        if obj is None: return
        for s in ndimage.__slots__:
            setattr(self, s,  getattr(obj, s, None))
            
    def __array_wrap__(self, out_arr, context=None):
        '''
        '''
        
        return numpy.ndarray.__array_wrap__(self, out_arr, context)

        
        