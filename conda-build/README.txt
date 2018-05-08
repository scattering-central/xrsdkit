# 1)
# start the build using the latest PyPI version: 
# > conda skeleton pypi xrsdkit 

# 2)
# move the resulting xrsdkit/meta.yaml 
# to this directory (conda-build/meta.yaml) 

# 2b)
# open meta.yaml and look for the build: tag.
# add noarch: python to the tag.
# the result should look like:
# build:
#   noarch: python

# 3)
# if this is the first build on the machine, 
# make a fresh conda environment
# > conda create -n xrsdkit python 
# ... then activate the environment
# > source activate xrsdkit 

# 3b) 
# install pymatgen
# > conda config --add channels matsci
# > conda install pymatgen 

# 4)
# invoke conda-build 
# > conda-build --user ssrl-paws conda-build/

# 5)
# make a free account on anaconda.org.
# install the anaconda client if it isn't already installed.
# > conda install anaconda-client

# 6) 
# log in using the anaconda client:
# > anaconda login

# 7)
# use the client to upload the package,
# using the output of step 4 above for package path
# (see the line marked "TEST END: <path-to-package>")
# NOTE: this may have happened automatically during conda-build,
# if you were already logged in
# > anaconda upload <path-to-package> 


