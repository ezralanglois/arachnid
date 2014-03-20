''' Check for certain dependencies

.. Created on Mar 20, 2014
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
from distutils.core import Command
from distutils import log
import sys

class check_dep(Command):
    ''' Check if the dependencies listed in `install_requires` and `extras_require`
    are currently installed and on the Python path.
    '''
    description = "Check if dependencies are installed"

    user_options = []
    test_commands = {}
    
    def initialize_options(self): pass
    def finalize_options(self): pass

    def run(self):
        ''' Check if dependencies are importable.
        '''
        
        packages = self.distribution.install_requires
        for v in self.distribution.extras_require.values():
            if isinstance(v, list): packages.extend(v)
            else: packages.append(v)
        sep=['>', '<', '>=', '<=', '==']
        found = []
        notfound=[]
        for package in packages:
            for s in sep:
                idx = package.find(s)
                if idx > -1: 
                    package=package[:idx]
                    break
            try:    mod = __import__(package)
            except: notfound.append(package)
            else:   found.append((package, mod))
        for package in notfound:
            log.info("Checking for %s: not found"%(package))
        log.info('---')
        for package, mod in found:
            version = ' - '+mod.__version__ if hasattr(mod, '__version__') else ''
            log.info("Checking for %s: found%s"%(package, version))
        sys.exit(len(notfound))
