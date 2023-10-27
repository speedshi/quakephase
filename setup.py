from setuptools import setup, find_packages

setup(
    name='quakephase',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'seisbench',
        'pyyaml',
        'scikit-learn',
        # Add any other dependencies here
    ],
)