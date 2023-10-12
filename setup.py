from setuptools import setup

setup(
    name='sparse-vd-keras',
    version='1.0',
    description='Sparse Variational Dropout layers for Keras Core/3.0',
    url='https://github.com/Faptimus420/Sparse_VD_keras-core',
    author='Patrik Zori',
    packages=['sparse_vd_keras'],
    install_requires=['keras_core>=0.1.7'],
    py_modules=['VariationalConv2d', 'VariationalDense']
)