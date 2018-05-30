from collections import OrderedDict

import numpy as np
import pypif.obj as pifobj

from . import profiler
from .. import structure_names
from .. import form_factor_names
from .. import crystalline_structure_names 

def make_pif(uid,expt_id=None,t_utc=None,q_I=None,temp_C=None,populations=None):
    """Make a pypif.obj.ChemicalSystem object describing XRSD data.

    Parameters
    ----------
    uid : str
        record id, should be unique across the dataset
    expt_id : str
        experiment id, should be shared across the experiment 
    t_utc : int
        UTC time in seconds
    q_I : array
        n-by-2 array of q (1/Angstrom) and intensity (arb)
    temp_C : float
        temperature of the sample in degrees C
    populations : dict
        dict defining sample populations and parameters 

    Returns
    -------
    csys : pypif.obj.ChemicalSystem
        pypif.obj.ChemicalSystem representation of the PIF record
    """
    csys = pifobj.ChemicalSystem()
    csys.uid = uid
    csys.ids = []
    csys.tags = []
    csys.properties = []
    if expt_id is not None:
        csys.ids.append(id_tag('EXPERIMENT_ID',expt_id))
    if t_utc is not None:
        csys.tags.append('time (utc): '+str(int(t_utc)))
    if q_I is not None:
        csys.properties.extend(q_I_properties(q_I,temp_C))
    if populations is not None:
        csys.properties.extend(populations_properties(populations))
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I))
    return csys

def id_tag(idname,idval,tags=None):
    return pifobj.Id(idname,idval,tags)

def q_I_properties(q_I,temp_C=None):
    properties = []
    # Process measured q_I into a property
    pI = q_I_property(q_I)
    if temp_C is not None:
        pI.conditions.append(pifobj.Value('temperature',
        [pifobj.Scalar(temp_C)],None,None,None,'degrees Celsius'))
    properties.append(pI)
    return properties

def q_I_property(q_I,qunits='1/Angstrom',Iunits='arb',propname='Intensity'):
    pI = pifobj.Property()
    n_qpoints = q_I.shape[0]
    pI.scalars = [pifobj.Scalar(q_I[i,1]) for i in range(n_qpoints)]
    pI.units = Iunits 
    pI.conditions = []
    pI.conditions.append( pifobj.Value('scattering vector magnitude', 
                        [pifobj.Scalar(q_I[i,0]) for i in range(n_qpoints)],
                        None,None,None,qunits) )
    pI.name = propname 
    return pI 

def profile_properties(q_I):
    prof = profiler.full_profile(q_I)
    props = []
    for fnm,fval in prof.items():
        if fval is not None:
            props.append(scalar_property(
            fnm,fval,'spectrum profiling quantity'))
    return props

def populations_properties(populations):
    properties = []
    for ip,popnm in enumerate(populations.keys()):
        properties.append(pop_name_property(ip,popnm))
    for ip,popnm in enumerate(populations.keys()):
        popd = populations[popnm]
        properties.append(pop_structure_property(popnm,popd))
        for ist, stnm in enumerate(popd['basis'].items()):
            stdef = popd['basis'][stnm]
            properties.append(site_name_property(popnm,ist,stnm)
            properties.append(site_ff_property(popnm,stnm,stdef)
    for ip,popnm in enumerate(populations.keys()):
        popd = populations[popnm]
        properties.extend(setting_properties(popnm,popd))
        properties.extend(param_properties(popnm,popd))
        for ist, stnm in enumerate(popd['basis'].items()):
            stdef = popd['basis'][stnm]
            properties.extend(site_setting_properties(popnm,stnm,stdef)
            properties.extend(site_param_properties(popnm,stnm,stdef)

def pop_name_property(ip,popnm):
    pp = pifobj.Property('population_{}_name'.format(ip))
    pp.tags = popnm
    return pp

def pop_structure_property(popnm,popd):
    pp = pifobj.Property('{}_structure'.format(popnm))
    pp.tags = popd['structure'] 
    return pp

def site_name_property(popnm,ist,stnm): 
    pp = pifobj.Property('{}_site_{}_name'.format(popnm,ist))
    pp.tags = stnm
    return pp

def site_ff_property(popnm,stnm,stdef): 
    pp = pifobj.Property('{}_{}_form'.format(popnm,stnm))
    pp.tags = stdef['form'] 
    return pp

def setting_properties(popnm,popd):
    pps = []
    for stgnm,stgval in popd['settings'].items():
        pp = pifobj.Property('{}_{}'.format(popnm,stgnm))
        pp.tags = str(stgval)
        pps.append(pp)
    return pps

def param_properties(popnm,popd):
    pps = []
    for pnm,pval in popd['parameters'].items():
        pp = pifobj.Property('{}_{}'.format(popnm,pnm),pval)
        pps.append(pp)
    return pps

def site_setting_properties(popnm,stnm,stdef):
    pps = []
    for stgnm,stgval in stdef['settings'].items():
        pp = pifobj.Property('{}_{}_{}'.format(popnm,stnm,stgnm))
        pp.tags = str(stgval)
        pps.append(pp)
    return pps

def site_param_properties(popnm,stnm,stdef):
    pps = []
    for pnm,pval in stdef['parameters'].items():
        pp = pifobj.Property('{}_{}_{}'.format(popnm,stnm,pnm),pval)
        pps.append(pp)
    return pps

def scalar_property(fname,fval,desc=None,data_type=None,funits=None):
    pf = pifobj.Property()
    pf.name = fname
    if isinstance(fval,list):
        pf.scalars = [pifobj.Scalar(v) for v in fval]
    else:
        pf.scalars = [pifobj.Scalar(fval)]
    if desc:
        pf.tags = [desc]
    if data_type:
        pf.dataType = data_type 
    if funits:
        pf.units = funits
    return pf


