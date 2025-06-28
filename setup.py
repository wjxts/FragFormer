#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import os
from setuptools import setup, find_packages, Extension
import sys

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fragment_mol.")

# if sys.platform == "darwin":
#     extra_compile_args = ["-stdlib=libc++", "-O3"]
# else:
#     extra_compile_args = ["-std=c++11", "-O3"]

extensions = [
    # Extension(
    #     "deepund.libbleu",
    #     sources=[
    #         "deepund/criterion/libbleu/libbleu.cpp",
    #         "deepund/criterion/libbleu/module.cpp",
    #     ],
    #     extra_compile_args=extra_compile_args,
    # ),
]
setup(
    name="fragment_mol",
    version="0.0.1",
    description="code base for MPP",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    setup_requires=[
        # "numpy",
        # "torch",
    ],
    # pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
    install_requires=[
        # "numpy",
        # "torch",
        # "matplotlib",
        # "seaborn",
        # "pandas",
        # 'juptyer',
        # "tqdm",
        # "wandb",
        # "warmup_scheduler",
        # "torchsummary",
        # "einops",
        # "hydra-core",
        # "omegaconf",
        # "scipy",
        # "sacremoses",
        # "sacrebleu==1.5.1",
        # "pre-commit",
        # 'black',  # only for develop
        # 'flake8' # only for develop
    ],
    packages=find_packages(include=["fragment_mol"]),
    ext_modules=extensions,
)