import app.setup
import util.setup
import pyspider.setup

gui_scripts = []
console_scripts = []
console_scripts.extend(["ara-"+script for script in app.setup.console_scripts])
console_scripts.extend(["ara-"+script for script in util.setup.console_scripts])
console_scripts.extend(["sp-"+script for script in pyspider.setup.console_scripts])

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('arachnid', parent_package, top_path)
    config.set_options(quiet=True)
    #config.add_subpackage('app')
    #config.add_subpackage('pyspider')
    #config.add_subpackage('util')
    config.add_subpackage('core')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())


