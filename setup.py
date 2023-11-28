from setuptools import setup, find_packages

setup(
    name='sparse-vd-keras',
    version='1.3',
    description='Sparse Variational Dropout layers for Keras 3.0',
    url='https://github.com/Faptimus420/Sparse_VD_keras-core',
    author='Cerphilly and Patrik Zori',
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    install_requires=['keras>=3.0'],
    python_requires='>=3.9',
)
