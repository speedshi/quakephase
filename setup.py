from setuptools import setup, find_packages

setup(
    name='quakephase',
    version='0.2.4',
    packages=find_packages(),
    author='Peidong Shi',
    author_email="speedshi@hotmail.com",
    description="A Python package for automatic seismic phase characterization",
    url="https://github.com/speedshi/quakephase",
    license="GPLv3",
    keywords="Seismology, Machine Learning, Phase Picking, Seismic Monitoring, Earthquake Monitoring",
    install_requires=[
        'seisbench',
        'pyyaml',
        'scikit-learn',
        # Add any other dependencies here
    ],
)