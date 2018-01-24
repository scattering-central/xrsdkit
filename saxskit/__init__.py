from collections import OrderedDict

population_keys = [\
    'unidentified',\
    'guinier_porod',\
    'spherical_normal',\
    'diffraction_peaks']

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

param_defaults = OrderedDict(
    I0_floor = 0.,
    G_gp = 1.,
    rg_gp = 10.,
    D_gp = 4.,
    I0_sphere = 1000.,
    r0_sphere = 20.,
    sigma_sphere = 0.05,
    q_pkcenter=0.1,
    I_pkcenter=1.,
    pk_hwhm = 0.001)

param_limits = OrderedDict(
    I0_floor = (0.,None),
    G_gp = (0.,None),
    rg_gp = (1.E-6,1.E4),
    D_gp = (0.,4.),
    I0_sphere = (0.,None),
    r0_sphere = (1.E-6,1.E4),
    sigma_sphere = (0.,0.5),
    q_pkcenter = (0.,1.),
    I_pkcenter = (0.,None),
    pk_hwhm = (1.E-6,0.1))

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
 

