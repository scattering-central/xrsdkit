from collections import OrderedDict

import numpy as np
import pypif.obj as pifobj

from . import saxs_fit, saxs_classify

def make_pif(uid,expt_id=None,t_utc=None,q_I=None,temp_C=None,flags=None,params=None,report=None):
    """Make a pypif.obj.ChemicalSystem object describing a SAXS experiment.

    Parameters
    ----------
    uid : str
        record id, should be unique across the record's destination dataset
    expt_id : str
        experiment id, should be the same as other records in the experiment
    t_utc : int
        UTC time in seconds
    q_I : array
        n-by-2 array of q (1/Angstrom) and intensity (arb)
    temp_C : float
        temperature of the sample in degrees C
    flags : dict
        dict of flags describing scatterer populations
    params : dict
        dict of parameters for computing saxs spectrum for flagged populations
    report : dict
        dict describing the fit objectives and SNR
        between measured `q_I` and the computed intensity from `params`

    Returns
    -------
    csys : pypif.obj.ChemicalSystem
        pypif.obj.ChemicalSystem representation of the PIF record
    """
    csys = pifobj.ChemicalSystem()
    csys.uid = uid
    csys.tags = []
    csys.ids = []
    if expt_id is not None:
        csys.ids.append(id_tag('EXPERIMENT_ID',expt_id))
    if t_utc is not None:
        csys.tags.append('time (utc): '+str(int(t_utc)))
    csys.properties = saxs_properties(q_I,temp_C,flags,params,report)
    return csys

def unpack_pif(pp):
    q_I = None
    temp = None
    flg = OrderedDict() 
    par = OrderedDict() 
    rpt = OrderedDict() 
    for prop in pp.properties:
        if prop.name == 'SAXS intensity':
            I = [float(sca.value) for sca in prop.scalars]
            for val in prop.conditions:
                if val.name == 'scattering vector':
                    q = [float(sca.value) for sca in val.scalars]
                if val.name == 'temperature':
                    temp = float(val.scalars[0].value)
            q_I = np.array(zip(q,I))
        elif prop.name[-5:] == '_flag' and prop.data_type == 'EXPERIMENTAL':
            flg[prop.name[:-5]] = bool(float(prop.scalars[0].value))
        elif prop.name in ['I0_sphere','r0_sphere','sigma_sphere',\
                        'G_precursor','rg_precursor',\
                        'I0_floor']:
            par[prop.name] = float(prop.scalars[0].value)
        elif prop.tags is not None:
            if 'spectrum fitting quantity' in prop.tags:
                rpt[prop.name] = float(prop.scalars[0].value)
    return q_I,temp,flg,par,rpt

def update_pif(pp,flags,params,report):
    q_I,temp_C,flg_old,par_old,rpt_old = unpack_pif(pp)
    csys = pifobj.ChemicalSystem()
    csys.uid = pp.uid
    csys.tags = pp.tags
    csys.ids = pp.ids
    csys.properties = saxs_properties(q_I,temp_C,flags,params,report)
    return csys

def saxs_properties(q_I,temp_C,flags,params,report):

    props = []

    if q_I is not None:
        # Process measured q_I into a property
        pI = q_I_property(q_I)
        if temp_C is not None:
            pI.conditions.append(pifobj.Value('temperature',
            [pifobj.Scalar(temp_C)],None,None,None,'degrees Celsius'))
        props.append(pI)

    if q_I is not None and params is not None and flags is not None:
        if not flags['bad_data'] and not flags['diffraction_peaks']:
            I_computed = saxs_fit.compute_saxs(q_I[:,0],flags,params)
            pI_computed = q_I_property(
                np.array([q_I[:,0],I_computed]).T,
                propname='computed SAXS intensity')
            props.append(pI_computed)

    if q_I is not None:
        # featurization of measured spectrum
        prof = saxs_fit.profile_spectrum(q_I)
        prof_props = profile_properties(prof)
        props.extend(prof_props)
        # ML flags for this featurization
        try:
            sxc = saxs_classify.SaxsClassifier()
            ml_flags = sxc.classify(np.array(list(prof.values())).reshape(1,-1))
            ml_flag_props = ml_flag_properties(ml_flags)
            props.extend(ml_flag_props)
        except:
            import pdb; pdb.set_trace()

    if flags is not None:
        fprops = ground_truth_flag_properties(flags)
        props.extend(fprops)
    if params is not None:
        pprops = param_properties(params)
        props.extend(pprops)
    if report is not None:
        rprops = fitreport_properties(report)
        props.extend(rprops)

    return props

def id_tag(idname,idval,tags=None):
    return pifobj.Id(idname,idval,tags)

def q_I_property(q_I,qunits='1/Angstrom',Iunits='arb',propname='SAXS intensity'):
    pI = pifobj.Property()
    n_qpoints = q_I.shape[0]
    pI.scalars = [pifobj.Scalar(q_I[i,1]) for i in range(n_qpoints)]
    pI.units = Iunits 
    pI.conditions = []
    pI.conditions.append( pifobj.Value('scattering vector', 
                        [pifobj.Scalar(q_I[i,0]) for i in range(n_qpoints)],
                        None,None,None,qunits) )
    pI.name = propname 
    return pI 

def profile_properties(prof):
    props = []
    for fnm,fval in prof.items():
        props.append(scalar_property(
        fnm,fval,'spectrum profiling quantity'))
    return props

def ml_flag_properties(ml_flags):
    props = []
    for fnm,fval in ml_flags.items():
        props.append(scalar_property(
            fnm+'_ML_flag',float(fval[0]),
            '{} ML flag'.format(fnm),'MACHINE_LEARNING'))
        props.append(scalar_property(
            fnm+'_ML_flag_prob',fval[1],
            '{} ML flag probability'.format(fnm),'MACHINE_LEARNING'))
    return props

def ground_truth_flag_properties(flags):
    props = []
    for fnm,fval in flags.items():
        props.append(scalar_property(
            fnm+'_flag',float(fval),
            '{} ground truth flag'.format(fnm),'EXPERIMENTAL'))
    return props

def param_properties(params):
    props = []
    if 'I0_floor' in params:
        props.append(scalar_property(
        'I0_floor',params['I0_floor'],
        'flat background intensity','FIT','arb'))
    if 'G_precursor' in params:
        props.append(scalar_property(
        'G_precursor',params['G_precursor'],
        'precursor Guinier factor','FIT','arb'))
    if 'rg_precursor' in params:
        props.append(scalar_property(
        'rg_precursor',params['rg_precursor'],
        'precursor radius of gyration','FIT','Angstrom'))
    if 'I0_sphere' in params:
        props.append(scalar_property(
        'I0_sphere',params['I0_sphere'],
        'spherical scatterer intensity','FIT','arb'))
    if 'r0_sphere' in params:
        props.append(scalar_property(
        'r0_sphere',params['r0_sphere'],
        'spherical scatterer mean radius','FIT','Angstrom'))
    if 'sigma_sphere' in params:
        props.append(scalar_property(
        'sigma_sphere',params['sigma_sphere'],
        'fractional standard deviation of spherical scatterer radii','FIT'))
    return props

def fitreport_properties(rpt):
    props = []
    print(rpt)
    for rptnm,rptval in rpt.items():
        #if isinstance(rptval,float):
        props.append(scalar_property(
        rptnm,rptval,'spectrum fitting quantity','FIT'))
    return props

def scalar_property(fname,fval,desc=None,data_type=None,funits=None):
    pf = pifobj.Property()
    pf.name = fname
    pf.scalars = [pifobj.Scalar(fval)]
    if desc:
        pf.tags = [desc]
    if data_type:
        pf.dataType = data_type 
    if funits:
        pf.units = funits
    return pf

