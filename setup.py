"""
    setup.py
    Author: Younhun Kim

    Package requirements are listed in `requirements.txt`.
"""

import os
import setuptools

lib_dir = os.path.dirname(os.path.realpath(__file__))

# Package requirements: Parse from `requirements.txt`.
requirementPath = lib_dir + '/requirements.txt'
requirements = []
if os.path.isfile(requirementPath):
    with open(requirementPath, "r") as f:
        requirements = f.read().splitlines()

setuptools.setup(
    name="chronostrain",
    version="0.0.1",
    author="Younhun Kim",
    author_email="younhun@mit.edu",
    url="https://github.com/gibsonlab/chronostrain",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: <TODO>",
        "Operating System :: Unix",
    ],
    packages=["chronostrain"],
    install_requires=requirements,
    python_requires='>=3.8',
)
