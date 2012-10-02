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
   :lines: 25-
   :linenos:
'''

if __name__ == '__main__':

    # Parameters
    
    resolution_file = "res_001.dat"
    freq_col = 'norm_freq'
    fsc_col = 'fsc'
    output_file = ''
    
    # Script
    
    
    from arachnid.core.metadata import format, format_utility
    import pylab, os
    
    # Read a resolution file
    
    res = format.read(resolution_file, numeric=True)
    res,header = format_utility.tuple2numpy(res)
    sfreq = header.index(freq_col)
    fsc = header.index(fsc_col)
    fsc = res[:, (sfreq, fsc)]
    
    # Plot FSC curve
        
    pylab.plot(fsc[:, 0], fsc[:, 1])
    pylab.axis([0.0,0.5, 0.0,1.0])
    pylab.xlabel('Spatial Frequency ($\AA^{-1}$)')
    pylab.ylabel('Fourier Shell Correlation')
    #pylab.title('Fourier Shell Correlation')
    
    if output_file == "":
        pylab.show()
    else:
        pylab.savefig(os.path.splitext(output_file)[0]+".png")


