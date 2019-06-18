#!/usr/bin/env python

from distutils.core import setup,Command

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'tests/runtests.py'])
        raise SystemExit(errno)

setup(name='pyppmi',
      version='0.1.0',
      description='Python Implementaion of Positive Pointwise Mutual Information Semantic Model',
      author='Kevin Brown',
      author_email='kevin.brown@oregonstate.edu',
      #url='https://github.com/thelahunginjeet/pyrankagg',
      packages=['pyppmi'],
      package_dir = {'pyppmi': ''},
      #package_data = {'pyrankagg' : ['tests/*.py']},
      cmdclass = {'test': PyTest},
      license='BSD-3',
      classifiers=[
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Programming Language :: Python',
          'Topic :: Statistics',
      ],
    )
