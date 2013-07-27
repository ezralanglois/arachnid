''' Utilities for QImages

.. Created on Jul 19, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from qt4_loader import QtGui,QtCore
import numpy

def qimage_to_html(qimg):
    '''
    '''
    
    ba = QtCore.QByteArray()
    buffer = QtCore.QBuffer(ba)
    buffer.open(QtCore.QIODevice.WriteOnly)
    qimg.save(buffer, 'PNG')
    return "<img src=\"data:img/png;base64,%s\">"%str(buffer.data().toBase64())

def _grayScaleColorModel(colortable=None):
    '''Create an RBG color table in gray scale
    
    :Parameters:
        
        colortable : list (optional)
                     Output list for QtGui.qRgb values
    :Returns:
        
        colortable : list (optional)
                     Output list for QtGui.qRgb values
    '''
    
    if colortable is None: colortable = []
    for i in xrange(256):
        colortable.append(QtGui.qRgb(i, i, i))
    return colortable
_basetable = _grayScaleColorModel()

def numpy_to_qimage(img, width=0, height=0, colortable=_basetable):
    ''' Convert a Numpy array to a PyQt4.QImage
    
    .. note::
        
        Some code adopted from: http://kogs-www.informatik.uni-hamburg.de/~meine/software/vigraqt/qimage2ndarray.py
    
    :Parameters:
    
    img : numpy.ndarray
          Array containing pixel data
    width : int
            Width of the image
    height : int
            Height of the image
    colortable : list
                 List of QtGui.qRgb values
    :Returns:
    
    qimg : PyQt4.QImage
           QImage representation
    '''
    
    if img.ndim == 3 and img.shape[2] in (3,4):
        bgra = numpy.empty((img.shape[0], img.shape[1], 4), numpy.uint8, 'C')
        bgra[...,0] = img[...,2]
        bgra[...,1] = img[...,1]
        bgra[...,2] = img[...,0]
        if img.shape[2] == 3:
            bgra[...,3].fill(255)
            fmt = QtGui.QImage.Format_RGB32
        else:
            bgra[...,3] = img[...,3]
            fmt = QtGui.QImage.Format_ARGB32
    
        qimage = QtGui.QImage(bgra.data, img.shape[1], img.shape[0], fmt)
        qimage._numpy = bgra
        return qimage
    elif img.ndim != 2: raise ValueError, "Only gray scale images are supported for conversion, %d"%img.ndim
    img = normalize_min_max(img, 0, 255.0, out=img)
    img = numpy.require(img, numpy.uint8, 'C')
    h, w = img.shape
    if width == 0: width = w
    if height == 0: height = h
    qimage = QtGui.QImage(img.data, width, height, width, QtGui.QImage.Format_Indexed8)
    qimage.setColorTable(colortable)
    qimage._numpy = img
    return qimage

def normalize_min_max(img, lower=0.0, upper=1.0, mask=None, out=None):
    ''' Normalize image to given lower and upper range
    
    :Parameters:
    
    img : numpy.ndarray
          Input image
    lower : float
            Lower value
    upper : numpy.ndarray
            Upper value
    mask : numpy.ndarray
           Mask for min/max calculation
    out : numpy.ndarray
          Output image
    
    :Returns:

    out : numpy.ndarray
          Normalized image
    '''
    
    if numpy.issubdtype(img.dtype, numpy.integer):
        img = img.astype(numpy.float)
    
    vmin = numpy.min(img) if mask is None else numpy.min(img*mask)
    out = numpy.subtract(img, vmin, out)
    vmax = numpy.max(img) if mask is None else numpy.max(img*mask)
    if vmax == 0: raise ValueError, "No information in image"
    numpy.divide(out, vmax, out)
    upper = upper-lower
    if upper != 1.0: numpy.multiply(upper, out, out)
    if lower != 0.0: numpy.add(lower, out, out)
    return out

def change_brightness(value, brightness):
    ''' Change pixel brightness by some factor
    '''
    
    return min(max(value + brightness * 255 / 100, 0), 255)

def change_contrast(value, contrast):
    ''' Change pixel contrast by some factor
    '''
    
    return min(max(((value-127) * contrast / 100) + 127, 0), 255)

def change_gamma(value, gamma):
    ''' Change pixel gamma by some factor
    '''
    
    return min(max(int( numpy.pow( value/255.0, 100.0/gamma ) * 255 ), 0), 255)

def adjust_level(func, colorTable, level):
    ''' Adjust the color level of an image
    
    :Parameters:
    
    func : function
           Adjustment function
    colorTable : list
                 List of QtGui.qRgb values
    level : int
            Current color level (0 - 255)
    
    :Returns:
    
    colorTable : list
                 List of QtGui.qRgb values
    '''
    
    table = []
    for color in colorTable:
        r = func(QtGui.qRed(color), level)
        g = func(QtGui.qGreen(color), level)
        b = func(QtGui.qBlue(color), level)
        table.append(QtGui.qRgb(r, g, b))
    return table

