from setuptools import setup

setup(
    name='sparse-vd-keras',
    version='1.0',
    description='Sparse Variational Dropout layers for Keras Core/3.0',
    url='https://github.com/Faptimus420/Sparse_VD_keras-core',
    author='Cerphilly and Patrik Zori',
    py_modules=['variationalconv2d', 'variationaldense'],
    install_requires=['keras_core>=0.1.7'],
    python_requires='>=3.9',
)
