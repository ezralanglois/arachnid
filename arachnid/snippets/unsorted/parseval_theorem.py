'''
.. Created on Oct 9, 2013
.. codeauthor:: robertlanglois
'''
import numpy
import scipy.fftpack

if __name__ == "__main__":
    
    if 1 == 0:
    
        img = numpy.random.rand(78,78)
        
        parseval_img = numpy.sum(img**2)
        
        parseval_fft=numpy.sum(numpy.abs(scipy.fftpack.fft2(img))**2) / img.ravel().shape[0]
    elif 1 == 1:
        img1 = numpy.random.rand(78,78)
        img2 = numpy.random.rand(78,78)
        diff = img1-img2
        
        parseval_img = numpy.sum(diff**2)
        parseval_fft=numpy.sum(numpy.abs(scipy.fftpack.fftshift(scipy.fftpack.fft2(diff)))**2) / img1.ravel().shape[0]
    elif 1 == 0:
        img1 = numpy.random.rand(78,78)
        img2 = numpy.random.rand(78,78)
        diff = img1-img2
        
        parseval_img = numpy.sum(diff**2)
        parseval_fft = (img1**2).sum() + (img2**2).sum() - 2*numpy.dot(img1.ravel(), img2[:, numpy.newaxis].ravel())
    else:
        img1 = numpy.random.rand(78,78)
        img2 = numpy.random.rand(78,78)
        diff = img1-img2
        parseval_img = numpy.sum(diff**2)
        
        fimg1=scipy.fftpack.fft2(img1)
        fimg2=scipy.fftpack.fft2(img2)
        parseval_fft =  (numpy.abs(fimg1)**2).sum() + (numpy.abs(fimg2)**2).sum() - 2*numpy.abs(numpy.dot(fimg1.ravel(), fimg2[:, numpy.newaxis].ravel()) )
        
        parseval_fft /= img1.ravel().shape[0]
    
    
    print parseval_img, '==', parseval_fft