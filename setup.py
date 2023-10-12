from setuptools import setup, find_packages

setup(
    name='sparse-vd-keras',
    version='1.0',
    description='Sparse Variational Dropout layers for Keras Core/3.0',
    url='https://github.com/Faptimus420/Sparse_VD_keras-core',
    author='Patrik Zori',
    packages=find_packages(exclude=['LeNet.py']),
    install_requires=['keras_core>=0.1.7'],
    python_requires='>=3.9',
)