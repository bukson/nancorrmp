from setuptools import setup
import sys

if not sys.version_info[0] == 3 and sys.version_info[1] < 6:
    sys.exit('Python < 3.6 is not supported')

version = '0.23'

setup(
    name='nancorrmp',
    packages=['nancorrmp', 'test'],
    version=version,
    description='A multiprocessing version of pandas correlation method',
    author='Michał Bukowski',
    author_email='michal.bukowski@buksoft.pl',
    license='MIT',
    url='https://github.com/bukson/nancorrmp',
    download_url='https://github.com/bukson/nancorrmp/tarball/' + version,
    keywords=['correlation', 'multiprocessing', 'pandas'],
    classifiers=[],
    install_requires=[
        'pandas',
        'numpy',
    ],
    tests_require=[
        'scipy'
    ],
)
