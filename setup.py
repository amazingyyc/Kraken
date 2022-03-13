# coding=utf-8

import os
import subprocess
import logging

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):

  def build_extension(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

    logging.info("extdir:", extdir)
    logging.info("ext.sourcedir:", ext.sourcedir)
    logging.info('self.build_temp:', self.build_temp)

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    subprocess.check_call(['cmake', ext.sourcedir, f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}'], cwd=self.build_temp)
    subprocess.check_call(['cmake', '--build', '.', '--config', 'Release', '-j16'], cwd=self.build_temp)


setup(
    name='kraken',
    version='0.0.2',
    author='amazingyyc',
    author_email='amazingyyc@outlook.com',
    zip_safe=False,
    packages=['kraken', 'kraken/pytorch'],
    ext_modules=[CMakeExtension('kraken')],
    cmdclass={'build_ext': CMakeBuildExt},
)
