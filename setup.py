from setuptools import setup, find_packages

setup(name='xrsdkit',
    version='0.0.13',
    url='https://github.com/scattering-central/xrsdkit.git',
    description='Scattering and diffraction analysis and modeling toolkit',
    author='SSRL',
    license='BSD',
    author_email='paws-developers@slac.stanford.edu',
    install_requires=['pyyaml','numpy','scipy','pandas','scikit-learn','lmfit','pymatgen'],
    packages=find_packages(),
    package_data={'xrsdkit':['scattering/*.yml','models/modeling_data/*.yml']}
)


