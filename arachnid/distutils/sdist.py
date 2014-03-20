''' Use GIT tags to define the version of the source distribution

.. note::
    
    Adopted from https://github.com/warner/python-ecdsa

.. Created on Mar 20, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from distutils.command import sdist as command_sdist
import subprocess
import os
import re

#####
VERSION_PY = """
# This file is originally generated from Git information by running 'setup.py
# sdist'. Distribution tarballs contain a pre-generated copy of this file.

__version__ = '%s'
"""

def update_version_py():
    '''
    Adopted from https://github.com/warner/python-ecdsa
    '''
    
    if not os.path.isdir(".git"):
        print "This does not appear to be a Git repository."
        return
    try:
        p = subprocess.Popen(["git", "describe",
                              "--tags"], #, "--dirty", "--always"
                             stdout=subprocess.PIPE)
    except EnvironmentError:
        print "unable to run git, leaving ecdsa/_version.py alone"
        return
    stdout = p.communicate()[0]
    if p.returncode != 0:
        print "unable to run git, leaving ecdsa/_version.py alone"
        return
    # we use tags like "v0.5", so strip the prefix
    assert stdout.startswith("v")
    ver = stdout[len("v"):].strip()
    # Ensure the version number is compatiable with eggs - Robert Langlois
    ver = ver.replace('-', '_') 
    f = open("arachnid/_version.py", "w")
    f.write(VERSION_PY % ver)
    f.close()
    print "set arachnid/_version.py to '%s'" % ver

def get_version():
    '''
    Adopted from https://github.com/warner/python-ecdsa
    '''
    
    try:
        f = open("arachnid/_version.py")
    except EnvironmentError:
        return None
    for line in f.readlines():
        mo = re.match("__version__ = '([^']+)'", line)
        if mo:
            ver = mo.group(1)
            n=ver.find('_')
            # Do not want to update every git commit
            if n  != -1:ver = ver[:n]
            return ver
    return None

class sdist(command_sdist.sdist):
    '''Adopted from https://github.com/warner/python-ecdsa
    '''
    def run(self):
        update_version_py()
        self.distribution.metadata.version = get_version()
        print 'Update version', self.distribution.metadata.version
        return command_sdist.sdist.run(self)
