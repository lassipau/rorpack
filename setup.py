from setuptools import setup, find_packages

setup(name='rorpack',
      version='1.0.0',
      description='Robust Output Regulation Package',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy>=1.0.0',
                        'matplotlib',
                        'slycot',
                        'control'],
      extras_require={
         'docs': ['sphinx'],
      },
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7'
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4'],
     )
