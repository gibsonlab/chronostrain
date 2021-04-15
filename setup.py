"""
    setup.py
    Author: Younhun Kim

    Package requirements are listed in `requirements.txt`.
"""

from pathlib import Path
import setuptools

lib_dir = Path(__file__).resolve().parent

# Package requirements: Parse from `requirements.txt`.
requirementPath = lib_dir / 'requirements.txt'
requirements = []
if Path(requirementPath).is_file():
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
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)
