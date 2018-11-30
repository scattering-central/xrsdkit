import copy
from collections import OrderedDict
import numpy as np
from .specie import Specie
from .. import definitions as xrsdefs 
from ..scattering import diffuse_intensity, disordered_intensity, crystalline_intensity
from ..scattering import space_groups as sgs

class StructureFormException(Exception):
    pass

class Population(object):

    def __init__(self,structure,settings={},parameters={},basis={}):
        # TODO: validate basis entries?
        self.structure = None
        self.settings = {}
        self.parameters = {}
        self.basis = {}
        self.set_structure(structure)
        self.update_basis(basis)
        self.update_settings(settings)
        self.update_parameters(parameters)

    def set_structure(self,structure):
        self.check_structure(structure,self.basis_to_dict())
        self.structure = structure
        # first update settings,
        # including any params associated with settings
        self.update_settings()
        # now update the rest of the params
        self.update_parameters()

    def set_form(self,specie_nm,form):
        if self.structure == 'crystalline' and form in xrsdefs.noncrystalline_form_factors:
            msg = 'structure {} does not support {} form factors'\
            .format(self.structure,form)
            raise StructureFormException(msg)
        else:
            self.basis[specie_nm].set_form(form)

    def update_settings(self,new_settings={}):
        current_stg_nms = list(self.settings.keys())
        # remove any non-valid settings
        for stg_nm in current_stg_nms:
            if not stg_nm in xrsdefs.structure_settings[self.structure]:
                self.settings.pop(stg_nm)
        # update settings, add any that are missing
        for stg_nm in xrsdefs.structure_settings[self.structure]:
            if stg_nm in new_settings:
                self.update_setting(stg_nm,new_settings[stg_nm])
            elif not stg_nm in self.settings:
                self.update_setting(stg_nm,xrsdefs.setting_defaults[stg_nm])

    def update_setting(self,stgnm,new_val):
        if stgnm == 'lattice':
            # ensure centering remains valid
            if ('centering' in self.settings) \
            and (not self.settings['centering'] in sgs.lattice_space_groups[new_val]):
                self.update_setting('centering','P')
        elif stgnm == 'centering':
            lat = self.settings['lattice']
            if not new_val in sgs.lattice_space_groups[lat]:
                raise ValueError('invalid centering {} for lattice {}'.format(new_val,lat))
        elif stgnm == 'space_group':
            lat = self.settings['lattice']
            cent = self.settings['centering']
            if new_val and not new_val in sgs.lattice_space_groups[lat][cent].values():
                raise ValueError('invalid space group {} for lattice {} and centering {}'.format(new_val,lat,cent))
            # TODO: if the space group is changed, check/update basis coordinates
        self.settings[stgnm] = new_val
        # if it is a lattice or interaction setting, update parameters 
        if stgnm in ['lattice','interaction']:
            self.update_parameters()

    def update_parameters(self,new_params={}):
        current_param_nms = list(self.parameters.keys())
        valid_param_nms = copy.deepcopy(xrsdefs.structure_params[self.structure])
        if self.structure == 'crystalline': 
            valid_param_nms.extend(copy.deepcopy(
            xrsdefs.setting_params['lattice'][self.settings['lattice']]))
        if self.structure == 'disordered': 
            valid_param_nms.extend(copy.deepcopy(
            xrsdefs.setting_params['interaction'][self.settings['interaction']]))
        # remove any non-valid params
        for param_nm in current_param_nms:
            if not param_nm in valid_param_nms:
                self.parameters.pop(param_nm)
        # add any missing params, taking from new_params if available 
        for param_nm in valid_param_nms:
            if not param_nm in self.parameters:
                self.parameters[param_nm] = copy.deepcopy(xrsdefs.param_defaults[param_nm]) 
            if param_nm in new_params:
                self.update_parameter(param_nm,new_params[param_nm])
           
    def update_parameter(self,param_nm,new_param_dict): 
        #if param_nm in self.parameters:
        #    if new_param_dict is not None:
        self.parameters[param_nm].update(new_param_dict)

    def add_specie(self,specie_name,ff_name,settings={},parameters={},coordinates=[]):
        if self.structure == 'crystalline' and ff_name in noncrystalline_form_factors:
            msg = 'structure {} does not support {} form factors'\
            .format(self.structure,ff_name)
            raise StructureFormException(msg) 
        self.basis[specie_name] = Specie(ff_name,settings,parameters,coordinates)

    def remove_specie(self,specie_name):
        # TODO: check for violated constraints
        # in absence of this specie? 
        self.basis.pop(specie_name)

    def to_dict(self):
        pd = {} 
        pd['structure'] = copy.copy(self.structure)
        pd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            pd['settings'][stg_nm] = copy.copy(stg)
        pd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            pd['parameters'][param_nm] = copy.deepcopy(param)
        pd['basis'] = self.basis_to_dict()
        return pd

    def to_ordered_dict(self):
        opd = OrderedDict()
        opd['structure'] = self.structure
        opd['settings'] = OrderedDict() 
        for stg_nm in xrsdefs.structure_settings[self.structure]:
            opd['settings'][stg_nm] = self.settings[stg_nm] 
        opd['parameters'] = OrderedDict() 
        for param_nm in xrsdefs.structure_parameters[self.structure]:
            opd['parameters'][param_nm] = self.parameters[param_nm].to_dict()
        opd['basis'] = self.basis_to_ordered_dict()
        return opd

    def basis_to_dict(self):
        bd = {}
        for specie_nm,specie in self.basis.items():
            bd[specie_nm] = specie.to_dict()
        return bd

    def basis_to_ordered_dict(self):
        obd = OrderedDict()
        for ffnm in xrsdefs.form_factor_names:
            for specie_nm,specd in popdef['basis'].items(): 
                # TODO: how should two species of the same form be ordered?
                if specd['form'] == ffnm:
                    obd[specie_nm] = copy.deepcopy(specd)
        return obd

    def update_from_dict(self,d):
        if 'basis' in d:
            self.update_basis(d['basis'])
        if 'structure' in d:
            self.set_structure(d['structure'])
        if 'settings' in d:
            self.update_settings(d['settings'])
        if 'parameters' in d:
            self.update_parameters(d['parameters'])

    def update_basis(self,bd):
        self.check_structure(self.structure,bd)
        for specie_nm,specd in bd.items():
            if specie_nm in self.basis:
                self.basis[specie_nm].update_from_dict(specd)
            else:
                self.basis[specie_nm] = Specie.from_dict(specd)

    @classmethod
    def from_dict(cls,d):
        inst = cls(d['structure'])
        inst.update_from_dict(d)
        return inst

    @staticmethod
    def check_structure(structure,basis):
        if structure == 'crystalline':
            for site_nm,specie_def in basis.items():
                if specie_def['form'] in xrsdefs.noncrystalline_form_factors:
                    msg = 'structure {} does not support {} form factors'\
                    .format(structure,specie_def['form'])
                    raise StructureFormException(msg)

    def compute_intensity(self,q,source_wavelength):
        if self.structure == 'diffuse':
            return diffuse_intensity(q,self.to_dict(),source_wavelength)
        elif self.structure == 'disordered':
            return disordered_intensity(q,self.to_dict(),source_wavelength)
        elif self.structure == 'crystalline':
            return crystalline_intensity(q,self.to_dict(),source_wavelength)
        elif self.structure == 'unidentified':
            return np.zeros(len(q))


