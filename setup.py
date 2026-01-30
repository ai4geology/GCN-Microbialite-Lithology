from setuptools import setup, find_packages

setup(
    name="gcn-lithology-identification",
    version="1.0.0",
    description="Spectral Graph Convolution Networks for Microbialite Lithology Identification",
    author="Keran Li, Jinmin Song et al.",
    author_email="keranli98@outlook.com",
    packages=find_packages(),
    install_requires=[
        'torch>=1.11.0',
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'imbalanced-learn>=0.8.0',
        'scipy>=1.7.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
    ],
)