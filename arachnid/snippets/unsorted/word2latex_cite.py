''' Converts word endnote citations (unformated) to latex citations

Download to edit and run: :download:`word2latex_cite.py <../../arachnid/snippets/word2latex_cite.py>`

To run:

.. sourcecode:: sh
    
    $ python word2latex_cite.py latex.tex latex_new.tex

.. note::
    
    You must have Arachnid and Matplotlib installed to run this script

.. literalinclude:: ../../arachnid/snippets/word2latex_cite.py
   :language: python
   :lines: 22-
   :linenos:
'''
import sys, re
sys.path.append('~/workspace/arachnida/src')
#sys.path.append('/guam.raid.home/robertl/tmp/arachnid-0.0.1')

if __name__ == '__main__':

    # Parameters
    input_file = sys.argv[1]
    bib_file = sys.argv[2]
    output_file = sys.argv[3]
    print input_file, '->', output_file, 'using', bib_file
    
    '''
    @article{Adiga2005,
   author = {Adiga, Umesh and Baxter, William T. and Hall, Richard J. and Rockel, Beate and Rath, Bimal K. and Frank, Joachim and Glaeser, Robert},
   title = {Particle picking by segmentation: A comparative study with SPIDER-based manual particle picking},
   journal = {Journal of Structural Biology},
   volume = {152},
   number = {3},
   pages = {211-220},
   note = {18},
    '''
    lines = open(bib_file, 'r').readlines()
    lib = "".join(lines)
    entries = lib.split('@')
    print len(entries)
    
    libmap={}
    for entry in entries[1:]:
        for cite in re.finditer(r'{([^\s]*),.*note\s.\s{([0-9]*)}', entry.replace('\n', ' ')):
            val, key = cite.groups()
            libmap[int(key)]=val
    
    fin = open(input_file, 'r')
    fout = open(output_file, 'w')
    
    for line in fin:
        #for cite in re.finditer(r'\s\{[^#]*#([^}]*)\}', line):
        for cite in re.finditer(r'\s\{([^}]*)\}', line):
            rec_list = [int(val[val.find('#')+1:]) for val in cite.groups()[0].split(';')]
            cite_list=[libmap[val] for val in rec_list]
            exp="".join(["[^#]*#%d"%rec for rec in rec_list])
            line = re.sub(r'\s\{%s\}'%exp, ' \cite{%s}'%(",".join(cite_list)), line)
            #sys.exit(0)
        fout.write(line)
    
    fin.close()
    fout.close()
    
