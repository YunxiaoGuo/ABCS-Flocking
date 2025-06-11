from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fixed_wing_model",
        ["fixed_wing_model.pyx"],
        extra_compile_args=["-O3"],
        extra_link_args=["-O3"],
        include_dirs=[np.get_include()]
    )
]

# Complies the .pyx file to .pyd
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",  # Enforce Python 3 syntax
            "boundscheck": False,  # Disable bounds checking (optional)
            "wraparound": False,  # Disable negative indexing (optional)
        },
        annotate=True
    )
)
# setup(ext_modules=extensions,compiler_directives={'language_level': "3",})
# Please execute "python setup.py build_ext --inplace --force" in cmd


