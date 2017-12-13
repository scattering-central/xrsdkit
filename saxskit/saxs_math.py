"""This module is for evaluating SAXS intensities and related quantities.

Computing the theoretical SAXS spectrum requires a dictionary of populations
(specifying the number of distinct groups of each scatterer type)
and a dictionary of parameters
(a set of parameters is expected for each population).

The supported populations and associated parameters are:

    - 'guinier_porod': scatterers described by the Guinier-Porod equations

      - 'G_gp': Guinier prefactor for Guinier-Porod scatterers 
      - 'rg_gp': radius of gyration for Guinier-Porod scatterers 
      - 'D_gp': Porod exponent for Guinier-Porod scatterers 

    - 'spherical_normal': populations of spheres 
        with normal (Gaussian) size distribution 

      - 'I0_sphere': spherical form factor scattering intensity scaling factor
      - 'r0_sphere': mean sphere size (Angstrom) 
      - 'sigma_sphere': fractional standard deviation of sphere size 

    - 'diffraction_peaks': Psuedo-Voigt diffraction peaks 

      - 'I_pkcenter': spherical form factor scattering intensity scaling factor
      - 'q_pkcenter': mean sphere size (Angstrom) 
      - 'pk_hwhm': fractional standard deviation of sphere size 

    - 'unidentified': if this population is indicated,
        then the scattering spectrum is unfamiliar. 
        This causes all other populations and parameters to be ignored.
        TODO: unidentified scattering could still yield a flat I(q=0).

    - Common parameters for all populations:
      
      - 'I0_floor': magnitude of noise floor, flat for all q. 

"""
from collections import OrderedDict

import numpy as np

from . import peakskit

# supported population types
population_keys = [\
    'unidentified',\
    'guinier_porod',\
    'spherical_normal',\
    'diffraction_peaks']

# features for profiling spectra
profile_keys = OrderedDict.fromkeys(population_keys)
profile_keys.update(dict(
    unidentified=[
        'Imax_over_Imean',
        'Imax_sharpness',
        'I_fluctuation',
        'logI_fluctuation',
        'logI_max_over_std',
        'r_fftIcentroid',
        'r_fftImax',
        'q_Icentroid',
        'q_logIcentroid',
        'pearson_q',
        'pearson_q2',
        'pearson_expq',
        'pearson_invexpq'],
    guinier_porod = [
        'I0_over_Imean',
        'I0_curvature',
        'q_at_half_I0'],
    spherical_normal=[
        'q_at_Iq4_min1',
        'pIq4_qwidth',
        'pI_qvertex',
        'pI_qwidth'],
    diffraction_peaks=[]))
all_profile_keys = []
for popk,profks in profile_keys.items():
    all_profile_keys.extend(profks)

# supported scattering parameters
parameter_keys = OrderedDict.fromkeys(population_keys)
parameter_keys.update(dict(
    unidentified = [
        'I0_floor'],
    guinier_porod = [
        'G_gp',
        'rg_gp',
        'D_gp'],
    spherical_normal = [
        'I0_sphere',
        'r0_sphere',
        'sigma_sphere'],
    diffraction_peaks = [
        'I_pkcenter',
        'q_pkcenter',
        'pk_hwhm']))
all_parameter_keys = []
for popk,parmks in parameter_keys.items():
    all_parameter_keys.extend(parmks)
 
def compute_saxs(q,populations,params):
    """Compute a SAXS intensity spectrum.

    TODO: Document the equation.

    Parameters
    ----------
    q : array
        Array of q values at which saxs intensity should be computed.
    populations : dict
        Each entry is an integer representing the number 
        of distinct populations of various types of scatterer. 
    params : dict
        Scattering equation parameters. 
        Each entry in the dict may be a float or a list of floats,
        depending on whether there are one or more of the corresponding
        scatterer populations.

    Returns
    ------- 
    I : array
        Array of scattering intensities for each of the input q values
    """
    I = np.zeros(len(q))
    if not bool(populations['unidentified']):
        n_gp = populations['guinier_porod']
        n_sph = populations['spherical_normal']
        n_pks = populations['diffraction_peaks']

        I0_floor = params['I0_floor'] 
        I = I0_floor*np.ones(len(q))

        if n_gp:
            rg_gp = params['rg_gp']
            G_gp = params['G_gp']
            D_gp = params['D_gp']
            for igp in range(n_gp):
                I_gp = guinier_porod(q,rg_gp[igp],D_gp[igp],G_gp[igp])
                I += I_gp

        if n_sph:
            I0_sph = params['I0_sphere']
            r0_sph = params['r0_sphere']
            sigma_sph = params['sigma_sphere']
            for isph in range(n_sph):
                I_sph = spherical_normal_saxs(q,r0_sph[isph],sigma_sph[isph])
                I += I0_sph[isph]*I_sph

        if n_pks:
            I_pk = params['I_pkcenter']
            q_pk = params['q_pkcenter']
            pk_hwhm = params['pk_hwhm']
        for ipk in range(n_pks):
            I_pseudovoigt = peakskit.peak_math.pseudo_voigt(q-q_pk[ipk],pk_hwhm,pk_hwhm)
            I += I_pk[ipk]*I_pseudovoigt

    return I

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

def spherical_normal_saxs(q,r0,sigma):
    """Compute SAXS intensity of a normally-distributed sphere population.

    The returned intensity is normalized 
    such that I(q=0) is equal to 1.
    The current version samples the distribution 
    from r0*(1-5*sigma) to r0*(1+5*sigma) 
    in steps of 0.02*sigma*r0.
    Originally contributed by Amanda Fournier.
    TODO: test distribution sampling, speed up if possible.

    Parameters
    ----------
    q : array
        array of scattering vector magnitudes
    r0 : float
        mean radius of the sphere population
    sigma : float
        fractional standard deviation of the sphere population radii

    Returns
    -------
    I : array
        Array of scattering intensities for each of the input q values
    """
    q_zero = (q == 0)
    q_nz = np.invert(q_zero) 
    I = np.zeros(q.shape)
    if sigma < 1E-9:
        x = q*r0
        V_r0 = float(4)/3*np.pi*r0**3
        I[q_nz] = V_r0**2 * (3.*(np.sin(x[q_nz])-x[q_nz]*np.cos(x[q_nz]))*x[q_nz]**-3)**2
        I_zero = V_r0**2 
    else:
        sigma_r = sigma*r0
        dr = sigma_r*0.02
        rmin = np.max([r0-5*sigma_r,dr])
        rmax = r0+5*sigma_r
        I_zero = 0
        for ri in np.arange(rmin,rmax,dr):
            xi = q*ri
            V_ri = float(4)/3*np.pi*ri**3
            # The normal-distributed density of particles with radius r_i:
            rhoi = 1./(np.sqrt(2*np.pi)*sigma_r)*np.exp(-1*(r0-ri)**2/(2*sigma_r**2))
            I_zero += V_ri**2 * rhoi*dr
            I[q_nz] += V_ri**2 * rhoi*dr*(3.*(np.sin(xi[q_nz])-xi[q_nz]*np.cos(xi[q_nz]))*xi[q_nz]**-3)**2
    if any(q_zero):
        I[q_zero] = I_zero
    I = I/I_zero 
    return I

def guinier_porod(q,r_g,porod_exponent,guinier_factor):
    """Compute the Guinier-Porod small-angle scattering intensity.
    
    Parameters
    ----------
    q : array
        array of q values
    r_g : float
        radius of gyration
    porod_exponent : float
        high-q Porod's law exponent
    guinier_factor : float
        low-q Guinier prefactor (equal to intensity at q=0)

    Returns
    -------
    I : array
        Array of scattering intensities for each of the input q values

    Reference
    ---------
    B. Hammouda, J. Appl. Cryst. (2010). 43, 716-719.
    """
    # q-domain boundary q_splice:
    q_splice = 1./r_g * np.sqrt(3./2*porod_exponent)
    idx_guinier = (q <= q_splice)
    idx_porod = (q > q_splice)
    # porod prefactor D:
    porod_factor = guinier_factor*np.exp(-1./2*porod_exponent)\
                    * (3./2*porod_exponent)**(1./2*porod_exponent)\
                    * 1./(r_g**porod_exponent)
    I = np.zeros(q.shape)
    # Guinier equation:
    if any(idx_guinier):
        I[idx_guinier] = guinier_factor * np.exp(-1./3*q[idx_guinier]**2*r_g**2)
    # Porod equation:
    if any(idx_porod):
        I[idx_porod] = porod_factor * 1./(q[idx_porod]**porod_exponent)
    return I

def fit_I0(q,I,order=4):
    """Find an estimate for I(q=0) by polynomial fitting.
    
    Parameters
    ----------
    q : array
        array of scattering vector magnitudes in 1/Angstrom
    I : array
        array of intensities corresponding to `q`

    Returns
    -------
    I_at_0 : float
        estimate of the intensity at q=0
    p_I0 : array
        polynomial coefficients for the polynomial 
        that was fit to obtain `I_at_0` (numpy format)
    """
    #TODO: add a sign constraint such that I(q=0) > 0?
    q_s,q_mean,q_std = standardize_array(q)
    I_s,I_mean,I_std = standardize_array(I)
    p_I0 = fit_with_slope_constraint(q_s,I_s,-1*q_mean/q_std,0,order) 
    I_at_0 = np.polyval(p_I0,-1*q_mean/q_std)*I_std+I_mean
    return I_at_0,p_I0

def standardize_array(data):
    d_mean = np.mean(data)
    d_std = np.std(data)
    d_s = (data-d_mean)/d_std
    return d_s,d_mean,d_std 

def fit_with_slope_constraint(q,I,q_cons,dIdq_cons,order,weights=None):
    """Fit scattering data to a polynomial with one slope constraint.

    This is performed by forming a Lagrangian 
    from a quadratic cost function 
    and the Lagrange-multiplied constraint function.
    Inputs q and I are not standardized in this function,
    so they should be standardized beforehand 
    if standardized fitting is desired.
    
    TODO: Document cost function, constraints, Lagrangian.

    Parameters
    ----------
    q : array
        array of scattering vector magnitudes in 1/Angstrom
    I : array
        array of intensities corresponding to `q`
    q_cons : float
        q-value at which a slope constraint will be enforced-
        because of the form of the Lagrangian,
        this constraint cannot be placed at exactly zero
        (it would result in indefinite matrix elements)
    dIdq_cons : float
        slope (dI/dq) that will be enforced at `q_cons`
    order : int
        order of the polynomial to fit
    weights : array
        array of weights for the fitting of `I`

    Returns
    -------
    p_fit : array
        polynomial coefficients for the fit of I(q) (numpy format)
    """
    Ap = np.zeros( (order+1,order+1),dtype=float )
    b = np.zeros(order+1,dtype=float)
    # TODO: vectorize the construction of Ap
    for i in range(0,order):
        for j in range(0,order):
            Ap[i,j] = np.sum( q**j * q**i )
        Ap[i,order] = -1*i*q_cons**(i-1)
    for j in range(0,order):
        Ap[order,j] = j*q_cons**(j-1)
        b[j] = np.sum(I*q**j)
    b[order] = dIdq_cons
    p_fit = np.linalg.solve(Ap,b) 
    p_fit = p_fit[:-1]  # throw away Lagrange multiplier term 
    p_fit = p_fit[::-1] # reverse coefs to get np.polyfit format
    return p_fit

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

def profile_spectrum(q_I):
    """Numerical profiling of a SAXS spectrum.

    Profile a saxs spectrum (n-by-2 array `q_I`) 
    by computing several relatively fast numerical metrics.
    The metrics should be invariant with respect to intensity scaling,
    and should not be strongly affected by minor details of the data
    (e.g. the limits of the reported q-range or the q-resolution).

    This method should execute gracefully
    for any n-by-2 input array,
    such that it can be used to profile any type of spectrum. 
    TODO: document the returned metrics here.

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q and scattered intensity I
    
    Returns
    -------
    features : dict
        Dictionary of metrics computed from input spectrum `q_I`.
        The features are:

        - 'Imax_over_Imean': maximum over mean intensity on the full q-range

        - 'Imax_sharpness': maximum over mean intensity for q-values 
            from 0.9*q(Imax) to 1.1*q(Imax)

        - 'I_fluctuation': sum of difference in I between adjacent points,
            multiplied by q-width of each point,
            divided by the intensity range (Imax minus Imin)

        - 'logI_fluctuation': same as I_fluctuation,
            but for log(I) and including only points where I>0

        - 'logI_max_over_std': max(log(I)) divided by std(log(I)),
            including only points with I>0

        - 'r_fftIcentroid': real-space centroid of the magnitude squared
            of the fourier transform of the scattering spectrum

        - 'r_fftImax': real-space maximum of the magnitude squared
            of the fourier transform of the scattering spectrum
 
        - 'q_Icentroid': q-space centroid of the scattering intensity

        - 'q_logIcentroid': q-space centroid of log(I)

        - 'pearson_q': Pearson correlation between q and I(q)

        - 'pearson_q2': Pearson correlation between q squared and I(q)

        - 'pearson_expq': Pearson correlation between exp(q) and I(q)
 
        - 'pearson_invexpq': Pearson correlation between exp(-q) and I(q) 
    """ 
    q = q_I[:,0]
    I = q_I[:,1]
    # I metrics
    idxmax = np.argmax(I)
    idxmin = np.argmin(I)
    I_min = I[idxmin]
    I_max = I[idxmax] 
    q_Imax = q[idxmax]
    I_range = I_max - I_min
    I_mean = np.mean(I)
    Imax_over_Imean = I_max/I_mean
    # log(I) metrics
    nz = I>0
    q_nz = q[nz]
    I_nz = I[nz]
    logI_nz = np.log(I_nz)
    logI_max = np.max(logI_nz)
    logI_min = np.min(logI_nz)
    logI_range = logI_max - logI_min
    logI_std = np.std(logI_nz)
    logI_max_over_std = logI_max / logI_std
    # I_max peak shape analysis
    idx_around_max = ((q > 0.9*q_Imax) & (q < 1.1*q_Imax))
    Imean_around_max = np.mean(I[idx_around_max])
    Imax_sharpness = I_max / Imean_around_max

    ### integration and intensity centroid
    dq = q[1:] - q[:-1]
    qcenter = 0.5 * (q[1:] + q[:-1])
    Itrap = 0.5 * (I[1:] + I[:-1])
    I_qint = np.sum(dq*Itrap)
    qI_qint = np.sum(qcenter*dq*Itrap)
    q_Icentroid = qI_qint / I_qint
    # same thing for log(I)
    dq_nz = q_nz[1:] - q_nz[:-1]
    qcenter_nz = 0.5 * (q_nz[1:] + q_nz[:-1])
    logItrap_nz = 0.5 * (logI_nz[1:] + logI_nz[:-1])
    logI_qint_nz = np.sum(dq_nz*logItrap_nz)
    qlogI_qint_nz = np.sum(qcenter_nz*dq_nz*logItrap_nz)
    q_logIcentroid = qlogI_qint_nz / logI_qint_nz

    ### fluctuation analysis
    nn_diff = I[1:]-I[:-1]
    I_fluctuation = np.sum(np.abs(nn_diff)*dq)/I_range
    nn_logdiff = logI_nz[1:]-logI_nz[:-1]
    logI_fluctuation = np.sum(np.abs(nn_logdiff)*dq_nz)/logI_range
    # keep indices where the sign of this difference changes.
    # also keep first index
    nn_diff_prod = nn_diff[1:]*nn_diff[:-1]
    idx_keep = np.hstack((np.array([True]),nn_diff_prod<0))
    fluc = np.sum(np.abs(nn_diff[idx_keep]))
    logI_fluctuation = fluc/logI_range

    ### correlation analysis
    pearson_q = compute_pearson(q,I)
    pearson_q2 = compute_pearson(q**2,I)
    pearson_expq = compute_pearson(np.exp(q),I)
    pearson_invexpq = compute_pearson(np.exp(-1*q),I)

    ### fourier analysis
    fftI = np.fft.fft(I)
    fftampI = np.abs(fftI)
    r = np.fft.fftfreq(q.shape[-1])
    idx_rpos = (r>0)
    r_pos = r[idx_rpos]
    fftampI_rpos = fftampI[idx_rpos]
    
    dr_pos = r_pos[1:] - r_pos[:-1]
    rcenter = 0.5 * (r_pos[1:] + r_pos[:-1])
    fftItrap = 0.5 * (fftampI_rpos[1:] + fftampI_rpos[:-1])
    fftI_rint = np.sum(dr_pos*fftItrap)
    rfftI_rint = np.sum(rcenter*dr_pos*fftItrap)
    r_fftIcentroid = rfftI_rint / fftI_rint 
    r_fftImax = r_pos[np.argmax(fftampI_rpos)]

    features = OrderedDict.fromkeys(profile_keys['unidentified'])
    features['Imax_over_Imean'] = Imax_over_Imean
    features['Imax_sharpness'] = Imax_sharpness
    features['I_fluctuation'] = I_fluctuation
    features['logI_fluctuation'] = logI_fluctuation
    features['logI_max_over_std'] = logI_max_over_std
    features['r_fftIcentroid'] = r_fftIcentroid
    features['r_fftImax'] = r_fftImax
    features['q_Icentroid'] = q_Icentroid
    features['q_logIcentroid'] = q_logIcentroid
    features['pearson_q'] = pearson_q 
    features['pearson_q2'] = pearson_q2
    features['pearson_expq'] = pearson_expq
    features['pearson_invexpq'] = pearson_invexpq
    return features 

def guinier_porod_profile(q_I):
    """Numerical profiling of guinier_porod scattering intensities.

    Computes the intensity at q=0 and the q value 
    at which the intensity drops to half of I(q=0).

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q and scattered intensity I

    Returns
    -------
    features : dict
        Dictionary of metrics computed from input spectrum `q_I`.
        The features are:

        - 'I0_over_Imean': intensity at q=0, obtained by polynomial fitting
            with the slope at q=0 constrained to be 0,
            divided by the average intensity. 

        - 'I0_curvature': curvature of the polynomial used in 'I0_over_Imean',
            evaluated at q=0, normalized by the mean intensity.

        - 'q_at_half_I0': q-value at which the intensity
            first drops to half of I(q=0)
    """
    q = q_I[:,0]
    I = q_I[:,1]
    features = OrderedDict.fromkeys(profile_keys['guinier_porod'])
    q_s,q_mean,q_std = standardize_array(q)
    I_s,I_mean,I_std = standardize_array(q)
    I_at_0, p_I0 = fit_I0(q,I,4)
    dpdq = np.polyder(p_I0)
    d2pdq2 = np.polyder(dpdq)
    I0_curv = (np.polyval(d2pdq2,-1*q_mean/q_std)*I_std+I_mean)/I_mean
    
    features['I0_over_Imean'] = I_at_0/I_mean
    idx_half_I0 = np.min(np.where(I<0.5*I_at_0))
    features['q_at_half_I0'] = q[idx_half_I0]
    features['I0_curvature'] = I0_curv 
    return features

def spherical_normal_profile(q_I):
    """Numerical profiling of spherical_normal scattering intensities.
    
    Computes several properties of the I(q) and I(q)*q**4 curves.

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q and scattered intensity I

    Returns
    -------
    features : dict
        Dictionary of metrics computed from input spectrum `q_I`.
        The features are:

        - 'q_at_Iqqqq_min1': q value at first minimum of I*q^4

        - 'pIqqqq_qwidth': Focal q-width of polynomial fit to I*q^4 
            near first minimum of I*q^4 

        - 'pI_qvertex': q value of vertex of polynomial fit to I(q) 
            near first minimum of I*q^4  

        - 'pI_qwidth': Focal q-width of polynomial fit to I(q) 
            near first minimum of I*q^4
    """
    q = q_I[:,0]
    I = q_I[:,1]
    features = OrderedDict.fromkeys(profile_keys['spherical_normal'])
    #######
    # 1: Find the first local max
    # and subsequent local minimum of I*q**4 
    Iqqqq = I*q**4
    # Window width for determining local extrema: 
    w = 10
    idxmax1, idxmin1 = 0,0
    stop_idx = len(q)-w-1
    test_range = np.arange(w,stop_idx)
    for idx in test_range:
        if np.argmax(Iqqqq[idx-w:idx+w+1]) == w and idxmax1 == 0:
            idxmax1 = idx
        if np.argmin(Iqqqq[idx-w:idx+w+1]) == w and idxmin1 == 0 and not idxmax1 == 0:
            idxmin1 = idx
    if idxmin1 == 0 or idxmax1 == 0:
        ex_msg = str('unable to find first maximum and minimum of I*q^4 '
        + 'by scanning for local extrema with a window width of {} points'.format(w))
        features['message'] = ex_msg 
        raise RuntimeError(ex_msg)
    #######
    # 2: Characterize I*q**4 around idxmin1. 
    idx_around_min1 = (q>0.9*q[idxmin1]) & (q<1.1*q[idxmin1])
    q_min1_mean = np.mean(q[idx_around_min1])
    q_min1_std = np.std(q[idx_around_min1])
    q_min1_s = (q[idx_around_min1]-q_min1_mean)/q_min1_std
    Iqqqq_min1_mean = np.mean(Iqqqq[idx_around_min1])
    Iqqqq_min1_std = np.std(Iqqqq[idx_around_min1])
    Iqqqq_min1_s = (Iqqqq[idx_around_min1]-Iqqqq_min1_mean)/Iqqqq_min1_std
    p_min1 = np.polyfit(q_min1_s,Iqqqq_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # quadratic vertex horizontal coord is -b/2a
    qs_at_min1 = -1*p_min1[1]/(2*p_min1[0])
    features['q_at_Iq4_min1'] = qs_at_min1*q_min1_std+q_min1_mean
    # quadratic focal width is 1/a 
    p_min1_fwidth = abs(1./p_min1[0])
    features['pIq4_qwidth'] = p_min1_fwidth*q_min1_std
    #######
    # 3: Characterize I(q) around idxmin1.
    I_min1_mean = np.mean(I[idx_around_min1])
    I_min1_std = np.std(I[idx_around_min1])
    I_min1_s = (I[idx_around_min1]-I_min1_mean)/I_min1_std
    pI_min1 = np.polyfit(q_min1_s,I_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # quadratic vertex horizontal coord is -b/2a
    qs_vertex = -1*pI_min1[1]/(2*pI_min1[0])
    features['pI_qvertex'] = qs_vertex*q_min1_std+q_min1_mean
    # quadratic focal width is 1/a 
    pI_fwidth = abs(1./pI_min1[0])
    features['pI_qwidth'] = pI_fwidth*q_min1_std
    return features 

def detailed_profile(q_I,populations):
    profs = OrderedDict()

    if bool(populations['unidentified']):
        return profs 

    #if bool(populations['guinier_porod']):
    try:
        gp_prof = guinier_porod_profile(q_I)
    except:
        gp_prof = OrderedDict.fromkeys(saxs_math.parameter_keys['guinier_porod'])
    profs.update(gp_prof)

    #if bool(populations['spherical_normal']):
    try:
        sph_prof = spherical_normal_profile(q_I)
    except:
        sph_prof = OrderedDict.fromkeys(saxs_math.parameter_keys['spherical_normal'])
    profs.update(sph_prof)

    #if bool(populations['diffraction_peaks']):
    #   diffraction-specific profiling should go here
 
    return profs


