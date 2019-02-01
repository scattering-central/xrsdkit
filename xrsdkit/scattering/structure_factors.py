import numpy as np

def hard_sphere_sf(q,r_sphere,volume_fraction):
    """Computes the Percus-Yevick hard-sphere structure factor. 

    Parameters
    ----------
    q : array
        array of q values 

    Returns
    -------
    F : float 
        Structure factor at `q`
    """
    p = volume_fraction
    d = 2*r_sphere
    qd = q*d
    qd2 = qd**2
    qd3 = qd**3
    qd4 = qd**4
    qd6 = qd**6
    sinqd = np.sin(qd)
    cosqd = np.cos(qd)
    l1 = (1+2*p)**2/(1-p)**4
    l2 = -1*(1+p/2)**2/(1-p)**4
    if q[0] == 0.:
        nc = np.zeros(len(q))
        nc[0] = -2*p*( 4*l1 + 18*p*l2 + p*l1 )
        nc[1:] = -24*p*(
            l1*( (sinqd[1:] - qd[1:]*cosqd[1:]) / qd3[1:] )
            -6*p*l2*( (qd2[1:]*cosqd[1:] - 2*qd[1:]*sinqd[1:] - 2*cosqd[1:] + 2) / qd4[1:] )
            -p*l1/2*( (qd4[1:]*cosqd[1:] - 4*qd3[1:]*sinqd[1:] - 12*qd2[1:]*cosqd[1:] 
                        + 24*qd[1:]*sinqd[1:] + 24*cosqd[1:] - 24) / qd6[1:] )
            )
    else:
        nc = -24*p*(
            l1*( (sinqd - qd*cosqd) / qd3 )
            -6*p*l2*( (qd2*cosqd - 2*qd*sinqd - 2*cosqd + 2) / qd4 )
            -p*l1/2*( (qd4*cosqd - 4*qd3*sinqd - 12*qd2*cosqd 
                        + 24*qd*sinqd + 24*cosqd - 24) / qd6 )
            )
    F = 1/(1-nc)
    return F

