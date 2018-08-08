"""This package provides tools for analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
a material System composed of one or more Populations,
each of which is composed of one or more Species.
This module outlines a taxonomy of classes and attributes
for describing and manipulating such a System.

Developer note: This is the only module that should require revision
when extending xrsdkit to new kinds of structures and form factors.
"""
from collections import OrderedDict

import numpy as np

from .population import Population
from .specie import Specie
from .. import * 

def structure_form_exception(structure,form):
    msg = 'structure specification {}'\
        'does not support specie specification {}- '\
        'this specie must be removed from the basis '\
        'before setting this structure'.format(structure,form)
    raise ValueError(msg)

class System(object):

    def __init__(self,populations={}):
        self.populations = populations
        self.fit_report = {}

    def to_dict(self):
        sd = {} 
        for pop_nm,pop in self.populations.items():
            sd[pop_nm] = pop.to_dict()
        return sd

    def to_ordered_dict(self):
        od = OrderedDict()
        ## Step 1: Standardize order of populations by structure and form,
        ## excluding entries for noise or unidentified structures
        for stnm in structure_names:
            for ffnm in form_factor_names:
                for pop_nm,pop in self.populations.items():
                    if not pop_nm == 'noise' \
                    and not pop.structure == 'unidentified':
                        if pop.structure == stnm \
                        and ffnm in [pop.basis[snm]['form'] for snm in pop.basis.keys()]:
                            ## Step 2: Standardize order of species by form factor
                            od[pop_nm] = pop.to_ordered_dict()
        ## Step 3: if noise or unidentified populations, put at the end
        for pop_nm,pop in self.populations.items():
            if pop_nm == 'noise' \
            or pop.structure == 'unidentified':
                od[pop_nm] = pop.to_ordered_dict() 
        return od

    def update_from_dict(self,d):
        for pop_name,pd_new in d.items():
            if not pop_name in self.populations:
                self.populations[pop_name] = Population.from_dict(pd_new) 

    @classmethod
    def from_dict(cls,d):
        inst = cls()
        inst.update_from_dict(d)
        return inst

    def compute_intensity(self,q,source_wavelength):
        """Computes scattering/diffraction intensity for some `q` values.

        TODO: Document the equations.

        Parameters
        ----------
        q : array
            Array of q values at which intensities will be computed
        source_wavelength : float 
            Wavelength of radiation source in Angstroms

        Returns
        ------- 
        I : array
            Array of scattering intensities for each of the input q values
        """
        I = np.zeros(len(q))
        for pop_name,pop in self.populations.items():
            I += pop.compute_intensity(q,source_wavelength)
        return I




# TODO: update convenience constructors to return System objects. 

def fcc_crystal(atom_symbol,a_lat=10.,pk_profile='voigt',I0=1.E-3,q_min=0.,q_max=1.,hwhm_g=0.001,hwhm_l=0.001):
    return dict(
        structure='fcc',
        settings={'q_min':q_min,'q_max':q_max,'profile':pk_profile},
        parameters={'I0':I0,'a':a_lat,'hwhm_g':hwhm_g,'hwhm_l':hwhm_l},
        basis={atom_symbol+'_atom':dict(
            coordinates=[0,0,0],
            form='atomic',
            settings={'symbol':atom_symbol}
            )}
        )

def unidentified_population():
    return dict(
        structure='unidentified',
        settings={}, 
        parameters={},
        basis={}
        )

def empty_site():
    return dict(
        form='diffuse',
        settings={},
        parameters={}
        )
        
def flat_noise(I0=1.E-3):
    return dict(
        structure='diffuse',
        settings={},
        parameters={'I0':I0},
        basis={'flat_noise':{'form':'flat'}}
        )


