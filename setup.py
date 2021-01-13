from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='ml_utils',
      packages=find_packages(),
      version="0.1.0",
      description='A collection of useful functions for ml projects',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/ml_utils.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm",
                         "psutil",
                         "opencv-python",
                         ],
      py_modules=['ml_utils'],
      long_description='''
            A collection of useful functions for ml projects
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
