from setuptools import setup, find_packages

setup(name='xrsdkit',
    version='0.0.8',
    url='https://github.com/scattering-central/xrsdkit.git',
    description='Scattering and diffraction analysis and modeling toolkit',
    author='SSRL',
    author_email='paws-developers@slac.stanford.edu',
    packages=find_packages(),
    install_requires=[
        'pyyaml','numpy','scipy','pandas','scikit-learn','lmfit','pymatgen'
    ],
    include_package_data=True,
    package_data={'xrsdkit':['scattering/atomic_scattering_params.yml','models/modeling_data/*.yml']}
)
