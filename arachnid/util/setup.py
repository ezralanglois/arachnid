''' Setup for utility scripts
'''

console_scripts = [
 'crop = arachnid.util.crop:main',
 'selrelion = arachnid.util.relion_selection:main',
 'bench = arachnid.util.bench:main',
 'enumfiles = arachnid.util.enumerate_filenames:main',
 'project = arachnid.util.project:main',
 'info = arachnid.util.image_info:main',
 'coverage = arachnid.util.coverage:main',
 'sanitycheck = arachnid.util.sanitycheck:sanitycheck',
 'screenmics = arachnid.util.screenmics:main',
 'delete = arachnid.util.delete:main',
 'prepvol = arachnid.util.prepvol:main',
]