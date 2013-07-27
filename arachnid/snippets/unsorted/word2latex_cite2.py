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
    print "Ref count:", len(entries)
    
    libmap={}
    for entry in entries[1:]:
        for cite in re.finditer(r'\{([^,-]+)', entry.replace('\n', ' ')):
            tag, = cite.groups()
            for cite in re.finditer(r'([^0-9]+)([1-2][0-9]{3})', tag):
                authordate = "".join(cite.groups())
                if authordate not in libmap: libmap[authordate]=[]
                libmap[authordate].append(tag)
                if len(libmap[authordate]) > 1:
                    print libmap[authordate]
                break
    print "Library entries", len(libmap) #, libmap.keys()
    fin = open(input_file, 'r')
    fout = open(output_file, 'w')
    
    total=0
    parano =0
    for line in fin:
        parano += 1
        newline=str(line)
        for cite in re.finditer(r'(\([A-Z][^\)]+ [1-2][0-9]{3}\))', line):
            if len(cite.groups()) == 0: continue
            cite = cite.groups()[0]
            n=newline.find(cite)
            origcite=cite
            cite = cite[1:len(cite)-1]
            if cite.find(';') != -1:
                cites = cite.split(';')
                newcite="\\cite{"
                authoryear=None
                ids=[]
                for cite in cites:
                    for ad in re.finditer(r'([^  -,]+).*([1-2][0-9]{3})', cite):
                        authoryear="".join(ad.groups()).lower()
                        break
                    try:
                        id='-'.join(libmap[authoryear])
                    except:
                        print 'Not found:', cite, '(', parano, ')', line[:20]
                        continue
                    ids.append(id)
                newcite+=",".join(ids)+"}"
                total+=1
                    
            else:
                authoryear=None
                for ad in re.finditer(r'([^  \-,]+).*([1-2][0-9]{3})', cite):
                     authoryear="".join(ad.groups()).lower()
                     break
                try:
                    id='-'.join(libmap[authoryear])
                    newcite="\\cite{%s}"%(id)
                except:
                    print 'Not found:', cite, '(', parano, ')', line[:20]
                    continue
                total+=1
            #print origcite, '->', newcite, '->', newline[n], newline[n+len(origcite)]
            newline = newline[:n]+newcite+newline[n+len(origcite):]
        fout.write(newline)
    print "total converted: ", total
    
    fin.close()
    fout.close()
    
