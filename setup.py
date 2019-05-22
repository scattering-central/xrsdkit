from setuptools import setup, find_packages

setup(name='xrsdkit',
    version='0.2.1',
    url='https://github.com/scattering-central/xrsdkit.git',
    description='Scattering and diffraction modeling and analysis toolkit',
    author='SSRL',
    license='BSD',
    author_email='paws-developers@slac.stanford.edu',
    install_requires=['pyyaml','numpy','scipy','pandas','scikit-learn<0.21.0','lmfit','matplotlib','dask_ml','paramiko'],
    packages=find_packages(),
    package_data={'xrsdkit':['scattering/*.yml']}
    )


