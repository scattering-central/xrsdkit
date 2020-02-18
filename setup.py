from setuptools import setup, find_packages

setup(name='xrsdkit',
    version='0.2.4',
    url='https://github.com/scattering-central/xrsdkit.git',
    description='Scattering and diffraction modeling and analysis toolkit',
    author='SSRL',
    license='BSD',
    author_email='paws-developers@slac.stanford.edu',
    install_requires=['pyyaml','numpy','scipy','pandas','scikit-learn','lmfit','matplotlib','dask_ml','dask[dataframe]','fsspec','paramiko'],
    packages=find_packages(),
    entry_points={'console_scripts':['xrsdkit-gui = xrsdkit.visualization.gui:run_gui']},
    package_data={'xrsdkit':['scattering/*.yml']}
    )


