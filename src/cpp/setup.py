"""Cython 编译脚本 — 将 .pyx 编译为 .pyd (Windows)"""
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="silver_guardian_cpp",
    ext_modules=cythonize(
        "pose_postprocess.pyx",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    include_dirs=[numpy.get_include()],
)
