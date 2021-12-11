"""
setup.py for ClassificaIO package
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description=f.read()

setup(
    name='LinkPridiction',
    packages=find_packages(),
    version='1.1.5.1',
    description='Graphical User Interface for machine learning classification algorithms from scikit-learn',
    long_description=long_description,
    include_package_data=True,
    author='Noam Keren & Dimitry yavestigneyev',
    author_email='nkrn1111@gmail.com',
    license='ort braude',
    url='https://github.com/dimaxer/DenseNet-LP',
    download_url='https://github.com/gmiaslab/ClassificaIO/archive/1.1.5.1.tar.gz',
    keywords=['machine learning', 'classification','Link Prediction'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Education',
        'Topic :: Utilities',
        ],
    install_requires=[
        'pandas>=0.23.3',
        'numpy>=1.21.4',
        'scipy>=1.1.0',
        'PyTorch>=1.0',
        'networks>=0.3.7',
        'fire>=0.4.0',
        'node2vec>=0.4.0'],
    zip_safe=False
)