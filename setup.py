from setuptools import setup, find_packages
import versioneer

from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

setup(
    name='WepyAnalysis',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Tools for analyzing weighted ensemble (WE) simulation data from Wepy',
    author='Ceren Kilinc, Samik Bose, Alex Dickson',
    author_email='kilincce@msu.com',
    license="MIT"
    classifiers=[
    	"Topic :: Utilities",
    	"License :: OSI Approved :: MIT License",
    	'Programming Language :: Python :: 3'
    ],

    #package

    packages=find_packages(where='src'),
    package_dir={'' : 'src'},

    include_package_data=True,
    # SNIPPET: this is a general way to do this
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],

    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'mdtraj',
        'wepy',
	    'csnanalysis',
	    'geomm',
        'scikit-learn',
        'deeptime',
	    'pickle',
    ],

)


