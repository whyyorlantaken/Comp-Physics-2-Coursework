# Import
from setuptools import setup, find_packages

# Setup configuration
setup(
    name = "atramenta",
    version = "0.1.0",
    description = "Simulation of orbits around a black hole",
    author = "Males-Araujo Yorlan",
    author_email = "yorlan.males@yachaytech.edu.ec",
    packages = find_packages(),
    install_requires = [
        "numpy",
        "matplotlib",
        "scipy",
        "h5py",
        "imageio",
        "IPython"
    ],
    python_requires = "==3.9.*",     # Version I used
)