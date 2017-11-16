from setuptools import setup, find_packages

setup(name='saxskit',
    version='0.0.3',
    url='https://github.com/scattering-central/saxskit.git',
    description='Data-driven SAXS analysis',
    author='SSRL',
    author_email='paws-developers@slac.stanford.edu',
    packages=find_packages(),
    install_requires=[
        'numpy','scipy','scikit-learn'
    ],
)
