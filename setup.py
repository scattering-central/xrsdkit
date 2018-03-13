from setuptools import setup, find_packages

setup(name='xrsdkit',
    version='0.0.5',
    url='https://github.com/scattering-central/xrsdkit.git',
    description='Scattering and diffraction analysis and modeling toolkit',
    author='SSRL',
    author_email='paws-developers@slac.stanford.edu',
    packages=find_packages(),
    install_requires=[
        'numpy','scipy','scikit-learn'
    ],
)
