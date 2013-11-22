''' Plots an FSC that can be customized for publication

See http://matplotlib.org/ for more details on customizing plotting.

Download to edit and run: :download:`plot_fsc.py <../../arachnid/snippets/plot_fsc.py>`

To run:

.. sourcecode:: sh
    
    $ python plot_fsc.py

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/plot_fsc.py
   :language: python
   :lines: 22-
   :linenos:
'''
from arachnid.core.metadata import format, format_utility
from arachnid.core.util import fitting
import pylab, os

if __name__ == '__main__':

    # Parameters
    
    resolution_file = os.path.expanduser("~/Desktop/dresraw_refine_005_mask_mask_raw_refine_005_fq6_sig13k3d0.tbi")
    freq_col = 'column1'
    fsc_col = 'column3'
    output_file = ''
    dpi=1000
    
    # Script
    
    
    # Read a resolution file
    
    res,header = format.read(resolution_file, ndarray=True)
    print "header from input resolution file: ", header
    sfreq = header.index(freq_col)
    fsc = header.index(fsc_col)
    fsc = res[:, (sfreq, fsc)]
    
    sp = 0.5
    if 1 == 0:
        coeff = fitting.fit_sigmoid(fsc[:, 0], fsc[:, 1])
        sp_0_5 = fitting.sigmoid_inv(coeff, sp)
    else:
        sp_0_5 = fitting.fit_linear_interp(fsc, sp)
    pylab.plot((fsc[0, 0], sp_0_5), (sp, sp), 'r--')
    pylab.plot((sp_0_5, sp_0_5), (0.0, sp), 'r--')
    pylab.text(sp_0_5+sp_0_5*0.1, sp, r'$%.2f \AA$'%(1.09/sp_0_5))
    
    
    sp = 0.143
    if 1 == 0:
        coeff = fitting.fit_sigmoid(fsc[:, 0], fsc[:, 1])
        sp_0_5 = fitting.sigmoid_inv(coeff, sp)
    else:
        sp_0_5 = fitting.fit_linear_interp(fsc, sp)
    pylab.plot((fsc[0, 0], sp_0_5), (sp, sp), 'b--')
    pylab.plot((sp_0_5, sp_0_5), (0.0, sp), 'b--')
    pylab.text(sp_0_5+sp_0_5*0.1, sp, r'$%.2f \AA$'%(1.09/sp_0_5))
    
    # Plot FSC curve
        
    pylab.plot(fsc[:, 0], fsc[:, 1], 'k-')
    pylab.axis([0.0,0.5, 0.0,1.0])
    pylab.xlabel('Spatial Frequency ($\AA^{-1}$)')
    pylab.ylabel('Fourier Shell Correlation')
    #pylab.title('Fourier Shell Correlation')
    
    if output_file == "":
        pylab.show()
    else:
        pylab.savefig(os.path.splitext(output_file)[0]+".png", dpi=dpi)


