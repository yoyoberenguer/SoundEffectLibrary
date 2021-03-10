# encoding: utf-8
# USE :
# python setup_project.py build_ext --inplace
#
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ext_modules = [Extension("SoundEffectLib", ["SoundEffectLib.pyx"],
               include_dirs=[numpy.get_include()], language="c++"),
              ]

#
# ext_modules = [
#                Extension("FadeEffect", ["FadeEffect.pyx"], include_dirs=[numpy.get_include()]),
#                Extension("Validation", ["Validation.pyx"], include_dirs=[numpy.get_include()])
#               ]

setup(
  name="SoundServer",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules
)

