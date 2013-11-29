''' Setup for core modules
'''


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy
    config = Configuration('core', parent_package, top_path)
    config.set_options(quiet=True)
    config.add_subpackage('image')
    config.add_subpackage('orient')
    config.add_subpackage('parallel')
    try: numpy_include = numpy.get_include()
    except: numpy_include = numpy.get_numpy_include() #@UndefinedVariable
    config.add_include_dirs(numpy_include)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
