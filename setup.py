from setuptools import setup, find_packages


setup(
    name='torchcrepe',
    version='0.0.0',
    url='https://github.com/maxrmorrison/torchcrepe',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    description='Pytorch implementation of CREPE pitch tracker',
    packages=find_packages(),
    install_requires=['librosa', 'torchaudio'])
