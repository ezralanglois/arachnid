''' Calibrate defocus pairs

Download to edit and run: :download:`calibrate_defocus.py <../../arachnid/snippets/calibrate_defocus.py>`

To run:

.. sourcecode:: sh
    
    $ python calibrate_defocus.py

.. literalinclude:: ../../arachnid/snippets/calibrate_defocus.py
   :language: python
   :lines: 16-
   :linenos:
'''
import numpy

if __name__ == '__main__':

    # Parameters
    
    output_file = "defocus_"
    defocus = '''38265.2
30118.1
56073.2
26707.5
33505.4
46350
28437
42813.3
20322.2
46005.4
34218.6
44040.8
30357.5
0
41072
24729.9
32065.3
19850.8
27330.4
13066
43921.9
34048.4
0
24826.7
32394.4
48325.4
22402.3
48812.8'''.split()
    defocus = [float(d) for d in defocus]
    
    diff='''8849.00
30971.50
-10618.80
-15485.75
-26547.00
-11061.25
-15485.75
17698.00
13273.50
15485.75
8406.55
-10618.80
-15928.20
-26989.45
    '''.split()
    diff = [float(d) for d in diff]
    
    defocus = numpy.asarray(defocus).reshape((len(defocus)/2, 2))
    estdiff = defocus[:, 0]-defocus[:, 1]
    diffdiff = (estdiff-diff)/2.0
    defocus_new = defocus.copy()
    defocus_new[:, 0]-=diffdiff
    defocus_new[:, 1]+=diffdiff
    print "\n".join([str(d) for d in defocus_new.ravel()])
    
    