# The RORPack project

The Robust Output Regulation Package is an open-source Python library for controller design and simulation for robut output tracking and disturbance rejection for linear partial differential equations. The package includes a set of examples (folder 'examples/') on simulation of different types of controlled PDE systems on 1D and 2D spatial domains.

## Requirements

The package works on Python 2 and Python 3, and it requires NumPy, MatPlotLib, and SciPy version greater than or equal to 1.0.0. Sphinx is an optional dependency which is used for generating the library documentation.

## Installation 

 1. Install Python 3, e.g., from the package manager (pkcon install python34 python34-tkinter)
 4. Install this library and its dependencies with the command 'pip install .' (or alternatively 'pip install -e .' for development). To install specifically for Python 3, it may be necessary to use 'pip' through the command option 'python3 -m', i.e.,  write all 'pip' commands in the form 'python3 -m pip install .' etc.
 5. Test to see that everything works by trying an example, e.g. 'python3 examples/wave_1d.py'

### Alternative: Installation using a Python virtual environment
 1. Install Python 3 from the package manager (pkcon install python34 python34-tkinter)
 2. Create a new virtual environment with the command 'python3 -m venv /path/to/new/virtual/environment'
 3. Activate the virtual environment with the command 'source /path/to/new/virtual/environment/bin/activate'
 4. Install this library and its dependencies with the command  'pip install .' (or alternatively 'pip install -e .' for development)
 5. Test to see that everything works by trying an example, e.g. 'python3 examples/wave_1d.py'

## Documentation

The mathematical documentation and general introduction to the package is contained in the folder ./"Introduction to RORPack".

The documentation for the source code (in html/MathJax format) is contained in the folder ./docs/build/html/

To generate the source code documentation::

    pip install . # Only necessary to do this once
    cd docs
    make html

## Disclaimer

The purpose of RORPack is to serve as a tool to illustrate the theory of robust output regulation for distributed parameter systems and it should not (yet) be considered as a serious controller design software. The developers of the software do not take any responsibility and are not liable for any damage caused through use of this software. In addition, at its present stage, the library is not optimized in terms of numerical accuracy. Any helpful comments on potential improvements of the numerical aspects will be greatly appreciated!


## Remarks

On Mac OS X it is possible that MatPlotLib might not work properly. To get around this, follow the discussion [here.](https://stackoverflow.com/questions/30280595/matplotlib-hangs-on-mac-osx-and-graph-is-not-displayed)

Some helpful remarks on differences between MATLAB and Python/NumPy/SciPy:
 1. https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
 2. http://mathesaurus.sourceforge.net/matlab-numpy.html
 3.1. In MATLAB reshape is different from reshape in NumPy, we are required to use fortran ordering in the latter to make things work
 3.2. In Python 1/2 is 0, in Matlab 1/2 is 0.5
 3.3. In MATLAB scalars are secretly 1x1 matrices, use numpy.atleast_2d to get the same effect in Python
 3.4. Conjugate transpose is A' in MATLAB, but in Python we must do A.conj().T
 3.5. Arrays in MATLAB are complex by default, but in numpy we need to specify it, e.g. a = np.array([[1]], 'complex')
 3.6. In NumPy only the real part of complex numbers is plotted automatically, unlike in MATLAB

## License

See LICENSE.txt for licensing information of the RORPack library.

