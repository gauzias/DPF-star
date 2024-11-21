# setup.py
from setuptools import setup, find_packages

setup(
    name="dpfstar",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'plotly',
        'scipy',
        'dash',
        'nibabel',
        'scipy',
        'numpy',
        'trimesh',
    ],
)