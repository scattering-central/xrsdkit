from collections import OrderedDict

from .specie import Specie
from .parameter import Parameter

from ..scattering import diffuse_intensity, disordered_intensity, crystalline_intensity

class Population(object):

    def __init__(self,structure,settings={},parameters={},basis={}):
        # TODO: validate basis entries?
        self.set_structure(structure)
        self.update_basis(basis)
        self.update_settings(settings)
        self.update_parameters(parameters)

    def set_structure(self,structure):
        self.check_structure(structure,self.basis_to_dict())
        new_settings = dict.fromkeys(structure_settings[structure])
        for stg_nm in structure_settings[structure]:
            if stg_nm in self.settings:
                new_settings[stg_nm] = self.settings[stg_nm]
            else:
                new_settings[stg_nm] = setting_defaults[stg_nm]
        self.settings = new_settings
        new_params = dict.fromkeys(structure_params[structure])  
        for param_nm in structure_params[structure]):
            if param_nm in self.parameters:
                new_params[param_nm] = self.parameters[param_nm]
            else:
                new_params[param_nm] = Parameter.from_dict(param_defaults[param_nm])
        self.parameters = new_params

    def add_specie(self,specie_name,ff_name,settings={},parameters={},coordinates=None):
        if self.structure in crystalline_structures and ff_name in noncrystalline_form_factors:
            structure_form_exception(self.structure,ff_name)
        self.basis[specie_name] = Specie(ff_name)
        self.basis[specie_name].update_settings(settings)
        self.basis[specie_name].update_parameters(parameters)
        self.basis[specie_name].update_coordinates(coordinates)

    def to_dict(self):
        pd = {} 
        pd['structure'] = self.structure
        pd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            pd['settings'][stg_nm] = stg 
        pd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            pd['parameters'][param_nm] = param.to_dict()
        pd['basis'] = self.basis_to_dict()
        return pd

    def to_ordered_dict(self):
        opd = OrderedDict()
        opd['structure'] = self.structure
        opd['settings'] = OrderedDict() 
        for stg_nm in structure_settings[self.structure]:
            opd['settings'][stg_nm] = self.settings[stg_nm] 
        opd['parameters'] = OrderedDict() 
        for param_nm in structure_parameters[self.structure]:
            opd['parameters'][param_nm] = self.parameters[param_nm].to_dict()
        opd['basis'] = self.basis_to_ordered_dict()
        return opd

    def basis_to_dict(self):
        bd = {}
        for specie_nm,specie in self.basis.items():
            pd['basis'][specie_nm] = specie.to_dict()
        return bd

    def basis_to_ordered_dict(self):
        obd = OrderedDict()
        for ffnm in form_factor_names:
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
        if 'parameters' in d:
            self.update_parameters(d['parameters'])
        if 'settings' in d:
            self.update_settings(d['settings'])

    def update_parameters(self,pd):
        for param_nm, paramd in pd.items():
            if param_nm in self.parameters:
                self.parameters[param_nm].update_from_dict(paramd)

    def update_settings(self,sd)
        for stg_nm, sval in sd.items():
            if stg_nm in self.settings:
                self.settings[stg_nm] = sval

    def update_basis(self,bd):
        self.check_structure(self.structure,bd)
            for specie_nm,specd in bd.items():
                self.basis[specie_nm] = Specie.from_dict(specd)

    @classmethod
    def from_dict(cls,d):
        inst = cls()
        inst.update_from_dict(d)
        return inst

    @staticmethod
    def check_structure(structure,basis):
        if structure in crystalline_structures:
            for site_nm,specie_def in basis.items():
                if specie_def['form'] in noncrystalline_form_factors:
                    structure_form_exception(structure,specie_def['form'])

    def compute_intensity(self,q,source_wavelength):
        if self.structure == 'diffuse':
            return diffuse_intensity(q,self.to_dict(),source_wavelength)
        elif self.structure == 'disordered':
            return disordered_intensity(q,self.to_dict(),source_wavelength)
        elif self.structure == 'crystalline':
            return crystalline_intensity(q,self.to_dict(),source_wavelength)
        elif self.structure == 'unidentified':
            return np.zeros(len(q))


