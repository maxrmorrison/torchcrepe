from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='torchcrepe',
    description='Pytorch implementation of CREPE pitch tracker',
    version='0.0.9',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torchcrepe',
    install_requires=['librosa', 'resampy', 'scipy', 'torch', 'tqdm'],
    packages=['torchcrepe'],
    package_data={'torchcrepe': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pitch', 'audio', 'speech', 'music', 'pytorch', 'crepe'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
