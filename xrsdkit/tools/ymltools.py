from collections import OrderedDict
import re
import copy
import pandas as pd

from . import profiler, primitives
from ..system import System

import yaml
import os, sys
import numpy as np
from xrsdkit.tools.profiler import profile_pattern
from .. import definitions as xrsdefs
from .piftools import _sort_populations, _sort_species

def save_sys_to_yaml(file_path,sys):
    sd = sys.to_dict()
    with open(file_path, 'w') as yaml_file:
        yaml.dump(primitives(sd),yaml_file)

def load_sys_from_yaml(file_path):
    with open(file_path, 'r') as yaml_file:
        sd = yaml.load(yaml_file)
    return System(**sd)

def setting_properties(ip,pop):
    """Create a dictionary of all settings with
    their values for a given population.

    Parameters
    ----------
    ip  : int
        population's number
    pop : xrsdkit.system.population.Population obj
        representation of a population

    Returns
    -------
    pps : dict
        dict of all settings with their values.
    """
    pps = {}
    for stgnm in xrsdefs.structure_settings[pop.structure]:
        stgval = xrsdefs.setting_defaults[stgnm]
        if stgnm in pop.settings:
            stgval = pop.settings[stgnm]
        pps['pop{}_{}'.format(ip,stgnm)] = str(stgval)
    return pps


def param_properties(ip,pop):
    """Create a dictionary of all parameters with
    their values for a given population.

    Parameters
    ----------
    ip  : int
        population's number
    pop : xrsdkit.system.population.Population obj
        representation of a population

    Returns
    -------
    pps : dict
        dict of all parameters with their values.
    """
    pps = {}
    param_nms = copy.deepcopy(xrsdefs.structure_params[pop.structure])
    if pop.structure == 'crystalline':
        param_nms.extend(xrsdefs.setting_params['lattice'][pop.settings['lattice']])
    if pop.structure == 'disordered':
        param_nms.extend(xrsdefs.setting_params['interaction'][pop.settings['interaction']])
    for param_nm in param_nms:
        pd = copy.deepcopy(xrsdefs.param_defaults[param_nm])
        if param_nm in pop.parameters:
            pd = pop.parameters[param_nm]
        pnm = 'pop{}_{}'.format(ip,param_nm)
        pps[pnm] = pd['value']
    return pps


def specie_setting_properties(ip,isp,specie):
    """Create a dictionary of all parameter with
    their values for given population.

    Parameters
    ----------
    ip  : int
        population's number
    isp : int
        specie's number
    pop : xrsdkit.system.specie.Specie obj
        representation of a specie

    Returns
    -------
    pps : dict
        dict of all settings with their values.
    """
    pps = {}
    for stgnm,stgval in specie.settings.items():
        pps['pop{}_specie{}_{}'.format(ip,isp,stgnm)] = str(stgval)
    return pps


def specie_param_properties(ip,isp,specie):
    """Create a dictionary of all parameters with
    their values for a given specie.

    Parameters
    ----------
    ip  : int
        population's number
    isp : int
        specie's number
    specie : xrsdkit.system.specie.Specie obj
        representation of a specie

    Returns
    -------
    pps : dict
        dict of all parameters with their values.
    """
    pps = {}
    for ic,cd in enumerate(specie.coordinates):
        if ic == 0: coord_id = 'x'
        if ic == 1: coord_id = 'y'
        if ic == 2: coord_id = 'z'
        pnm = 'pop{}_specie{}_coord{}'.format(ip,isp,coord_id)
        pps[pnm] = cd['value']
    for param_nm,pd in specie.parameters.items():
        pnm = 'pop{}_specie{}_{}'.format(ip,isp,param_nm)
        pps[pnm] = pd['value']
    return pps


def get_data_from_local_dir(path_to_dir,output_csv=False):
    """Get data from a local directory.

    The data directory should contain one or more subdirectories,
    where each subdirectory contains .yml files,
    where each .yml file describes one sample,
    as saved by save_sys_to_yaml().
    If `output_csv` is True,
    the dataset is saved to dataset.csv in `path_to_dir`.

    Parameters
    ----------
    path_to_dir : str
        absolute path to the folder with the training set
        Precondition: dataset directory includes directories named
        by the name of the experiments; each experiment directory
        holds data from this experiment. For each sample there is one yml file
        with System object ... and one dat file with q and I arrays
        (array of scattering vector magnitudes, array of integrated
        scattering intensities).

    Returns
    -------
    df_work : pandas.DataFrame
        dataframe containing features and labels
        exctracted from the dataset.
    """
    data = []
    reg_labels = []
    cls_labels = []
    all_reg_labels = set()  # all_reg_labels and all_cls_labels are sets of all
    all_cls_labels = set()  # unique labels over the provided datasets

    print('reading dataset from {}'.format(path_to_dir))
    samples = read_data(path_to_dir)
    print('done - found {} records'.format(len(samples)))

    for pp in samples:
        expt_id, sample_id, features, classification_labels, regression_outputs = \
            unpack_sample(pp, path_to_dir)

        for k,v in regression_outputs.items():
            all_reg_labels.add(k)
        reg_labels.append(regression_outputs)

        for k,v in classification_labels.items():
            all_cls_labels.add(k)
        cls_labels.append(classification_labels)

        data_row = [expt_id] + [sample_id] + list(features.values())
        data.append(data_row)

    reg_labels_list = list(all_reg_labels)
    reg_labels_list.sort()

    cls_labels_list = list(all_cls_labels)
    cls_labels_list.sort()

    for datai,rli,cli in zip(data,reg_labels,cls_labels):
        orl = OrderedDict.fromkeys(reg_labels_list)
        ocl = OrderedDict.fromkeys(cls_labels_list)
        orl.update(rli)
        ocl.update(cli)
        datai.extend(list(orl.values()))
        datai.extend(list(ocl.values()))

    colnames = ['experiment_id'] + ['sample_id'] +\
            copy.copy(profiler.profile_keys) + \
            reg_labels_list + \
            cls_labels_list

    df_work = pd.DataFrame(data=data, columns=colnames)
    if output_csv:
        df_work.to_csv(os.path.join(path_to_dir,'dataset.csv'))
    return df_work

def read_data(path_to_dir):
    """Load content of yml files from provided directory.

    Parameters
    ----------
    path_to_dir : str
        abs path to the folder with the training set

    Returns
    -------
    samples : list
        list of dictionaries loaded from yml files that include
        fit_report, description of all populations, and sample_metadata.
    """
    
    samples = []
    for experiment in os.listdir(path_to_dir):
        exp_data_dir = os.path.join(path_to_dir,experiment)
        if os.path.isdir(exp_data_dir):
            for s_data_file in os.listdir(exp_data_dir):
                if s_data_file.endswith('.yml'):
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    sys = load_sys_from_yaml(file_path)
                    if bool(int(sys.fit_report['good_fit'])):
                        samples.append(sys.to_dict())
    return samples

def unpack_sample(pp, path_to_dir):
    """Extract features and labels from the dict describing the sample.

    Parameters
    ----------
    pp : dict
        dict containing description of xrsdkit.system.System.
        Includes fit_report, sample_metadata, features,
        noise_model, and one dict for each of the populations.
    path_to_dir : str
        abs path to the folder with the training set

    Returns
    -------
    expt_id : str
        name of the experiment (must be unique for a training set)
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    classification_labels : dict
        dict of all classification labels with their values for given sample
    regression_labels : dict
        dict of all regression labels with their values for given sample
    sample_id : str
        uinque sample id (includes experiment id)
    """
    expt_id = pp['sample_metadata']['experiment_id']
    sample_id = pp['sample_metadata']['sample_id']
    features = pp['features']
    #file_path = os.path.join(path_to_dir,expt_id,pp['sample_metadata']['data_file'])
    #q_I = np.loadtxt(file_path)
    #q = q_I[:,0]
    #I = q_I[:,1]
    #features = profile_pattern(q, I)
    sys = System(**pp)

    regression_labels = {}
    classification_labels = {}
    sys_cls = ''
    ipop = 0

    I0 = sys.noise_model.parameters['I0']['value']
    for k, v in sys.populations.items():
        I0 += v.parameters['I0']['value']

    I0_noise = sys.noise_model.parameters['I0']['value']
    if I0 == 0.:
        regression_labels['noise_I0_fraction'] = 0.
    else:
        regression_labels['noise_I0_fraction'] = I0_noise/I0

    classification_labels['noise_classification'] =sys.noise_model.model
    for param_nm,pd in sys.noise_model.parameters.items():
        regression_labels['noise_'+param_nm] = pd['value']
    for struct_nm in xrsdefs.structure_names: # use structure_names to impose order
        struct_pops = OrderedDict()
        for pop_nm,pop in sys.populations.items():
            if pop.structure == struct_nm:
                struct_pops[pop_nm] = pop
        # sort any populations with same structure
        struct_pops = _sort_populations(struct_nm,struct_pops)
        for pop_nm, pop in struct_pops.items():
            if I0 == 0.:
                regression_labels['pop{}_I0_fraction'.format(ipop)] = 0.
            else:
                pop_I0 = pop.parameters['I0']['value']
                regression_labels['pop{}_I0_fraction'.format(ipop)] = pop_I0/I0
            regression_labels.update(param_properties(ipop,pop))
            classification_labels.update(setting_properties(ipop,pop))
            if sys_cls: sys_cls += '__'
            sys_cls += 'pop{}_{}'.format(ipop,pop.structure)
            #print(pop.structure)
            classification_labels['pop{}_structure'.format(ipop)] = pop.structure
            bas_cls = ''
            ispec = 0
            for ff_nm in xrsdefs.form_factor_names: # use form_factor_names to impose order
                ff_species = OrderedDict()
                for specie_nm,specie in pop.basis.items():
                    if specie.form == ff_nm:
                        ff_species[specie_nm] = specie
                # sort any species with same form
                ff_species = _sort_species(ff_nm,ff_species)
                for specie_nm,specie in ff_species.items():
                    regression_labels.update(specie_param_properties(ipop,ispec,specie))
                    classification_labels.update(specie_setting_properties(ipop,ispec,specie))
                    if bas_cls: bas_cls += '__'
                    bas_cls += 'specie{}_{}'.format(ispec,specie.form)
                    classification_labels['pop{}_specie{}_form'.format(ipop,ispec)] =specie.form
                    ispec += 1
            classification_labels['pop{}_basis_classification'.format(ipop)] = bas_cls

            ipop += 1

    if sys_cls == '':
        sys_cls = 'unidentified'
    classification_labels['system_classification'] =sys_cls
    return expt_id, sample_id, features, classification_labels, regression_labels
