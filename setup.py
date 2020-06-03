from setuptools import setup, find_packages


setup(
    name='torchcrepe',
    description='Pytorch implementation of CREPE pitch tracker',
    version='0.0.0',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torchcrepe',
    install_requires=['librosa', 'torchaudio'],
    packages=find_packages(),
    include_package_data=True)
