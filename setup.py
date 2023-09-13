from setuptools import setup, find_packages

setup(
    name='quakephase',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'seisbench',
        'pyyaml',
        # Add any other dependencies here
    ],
)