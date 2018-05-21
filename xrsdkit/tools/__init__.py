import numpy as np
import yaml

def save_fit(file_path,populations,fixed_params,param_bounds,param_constraints,report):
    with open(file_path, 'w') as yaml_file:
        yaml.dump({'populations':primitives(populations),
            'fixed_params':primitives(fixed_params),
            'param_bounds':primitives(param_bounds),
            'param_constraints':primitives(param_constraints),
            'report':primitives(report)},yaml_file)

def load_fit(file_path):
    with open(file_path, 'r') as yaml_file:
        data = yaml.load()
    return data['populations'],data['fixed_params'],data['param_bounds'],data['param_constraints'],data['report']

def primitives(v):
    if isinstance(v,dict):
        rd = {}
        for kk,vv in v.items():
            rd[kk] = primitives(vv)
        return rd
    elif isinstance(v,list):
        return [primitives(vv) for vv in v]
    elif isinstance(v,str):
        return str(v)
    elif isinstance(v,int):
        return int(v)
    else:
        return float(v)

def standardize_array(x):
    xmean = np.mean(x)
    xstd = np.std(x)
    xs = (x-xmean)/xstd
    return xs, xmean, xstd

def g_of_r(q_I):
    """Compute g(r) and the maximum characteristic scatterer length.

    Parameters
    ----------
    q_I : array
        n-by-2 array of q values and intensities

    Returns
    -------
    g_of_r : array
        n-by-2 array of r values and g(r) magnitudes
    r_max : float
        maximum scatterer length- the integral of g(r) from zero to r_max
        is 0.99 times the full integral of g(r)
    """
    q = q_I[:,0]
    I = q_I[:,1]
    fftI = np.fft.fft(I)
    fftampI = np.abs(fftI)
    r = np.fft.fftfreq(q.shape[-1])
    idx_rpos = (r>=0)
    r_pos = r[idx_rpos]
    nr_pos = len(r_pos)
    fftampI_rpos = fftampI[idx_rpos]
    dr_pos = r_pos[1:]-r_pos[:-1] 
    fftI_trapz = (fftampI_rpos[1:]+fftampI_rpos[:-1])*0.5 
    fftI_dr = dr_pos * fftI_trapz
    fft_tot = np.sum(fftI_dr)
    idx = 0
    fftsum = fftI_dr[idx]
    while fftsum < 0.99*fft_tot:
        idx += 1
        fftsum += fftI_dr[idx]
    r_max = r_pos[idx]
    return np.vstack([r_pos,fftampI_rpos]).T,r_max

def Rsquared(y1,y2):
    """Compute the coefficient of determination.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats

    Returns
    -------
    Rsquared : float
        coefficient of determination between `y1` and `y2`
    """
    sum_var = np.sum( (y1-np.mean(y1))**2 )
    sum_res = np.sum( (y1-y2)**2 ) 
    return float(1)-float(sum_res)/sum_var

def pearson(y1,y2):
    """Compute the Pearson correlation coefficient.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats

    Returns
    -------
    pearson_r : float
        Pearson's correlation coefficient between `y1` and `y2`
    """
    y1mean = np.mean(y1)
    y2mean = np.mean(y2)
    y1std = np.std(y1)
    y2std = np.std(y2)
    return np.sum((y1-y1mean)*(y2-y2mean))/(np.sqrt(np.sum((y1-y1mean)**2))*np.sqrt(np.sum((y2-y2mean)**2)))

def compute_chi2(y1,y2,weights=None):
    """Compute sum of difference squared between two arrays.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats
    weights : array
        array of weights to multiply each element of (`y2`-`y1`)**2 

    Returns
    -------
    chi2 : float
        sum of difference squared between `y1` and `y2`. 
    """
    if weights is None:
        return np.sum( (y1 - y2)**2 )
    else:
        weights = weights / np.sum(weights)
        return np.sum( (y1 - y2)**2*weights )

def compute_pearson(y1,y2):
    """Compute the Pearson correlation coefficient.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats

    Returns
    -------
    pearson_r : float
        Pearson's correlation coefficient between `y1` and `y2`
    """
    y1mean = np.mean(y1)
    y2mean = np.mean(y2)
    y1std = np.std(y1)
    y2std = np.std(y2)
    return np.sum((y1-y1mean)*(y2-y2mean))/(np.sqrt(np.sum((y1-y1mean)**2))*np.sqrt(np.sum((y2-y2mean)**2)))

def compute_Rsquared(y1,y2):
    """Compute the coefficient of determination.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats

    Returns
    -------
    Rsquared : float
        coefficient of determination between `y1` and `y2`
    """
    sum_var = np.sum( (y1-np.mean(y1))**2 )
    sum_res = np.sum( (y1-y2)**2 ) 
    return float(1)-float(sum_res)/sum_var



