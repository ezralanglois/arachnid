''' Setup for utility scripts
'''

console_scripts = [
 'crop = arachnid.util.crop:main',
 'selrelion = arachnid.util.relion_selection:main',
 'bench = arachnid.util.bench:main',
 'convert2spi = arachnid.util.spider_prep:main',
 'micrographsel = arachnid.util.mic_select:main',
 'restack = arachnid.util.restack:main',
 'screenmics = arachnid.util.screen_mics:main',
 'frameavg = arachnid.util.average_frame:main',
 'info = arachnid.util.image_info:main',
 'relion-stack = arachnid.util.relion_align_stack:main',
]