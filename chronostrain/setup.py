"""
    setup.py
    Author: Younhun Kim

    Package requirements are listed in `requirements.txt`.
"""

import os
import setuptools

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
requirements = []
if os.path.isfile(requirementPath):
    with open(requirementPath, "r") as f:
        requirements = f.read().splitlines()

setuptools.setup(
    name="chronostrain",
    version="0.0.1",
    author="Younhun Kim",
    author_email="younhun@mit.edu",
    url="https://github.com/gibson",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)
