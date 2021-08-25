from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(name="c_routines", 
    	      sources=["c_routines.pyx"],
              # include_dirs=[...],
              # libraries=[...],
              # library_dirs=[...],
              extra_compile_args=['-ffast-math','-O3'],
              language='c++'),
]

setup(
    name="CircuitModels",
    ext_modules=cythonize(extensions,annotate=True,compiler_directives={'language_level' : "3"}),
)
