"""xrsdkit: a package for data-driven scattering and diffraction analysis"""
import os
from sys import platform as sys_pf
import matplotlib
if 'DISPLAY' in os.environ or sys_pf == 'darwin':
    matplotlib.use("TkAgg")
# API will be defined here

