import copy

import numpy as np

from . import symmetries
from . import space_groups as sgs

def get_lattice_vectors(lattice_id,a=None,b=None,c=None,alpha=None,beta=None,gamma=None):
    # TODO: expand to support all lattices 
    if lattice_id in ['cubic']:
        a1 = [a, 0., 0.]
        a2 = [0., a, 0.]
        a3 = [0., 0., a]
    elif lattice_id in ['hexagonal']:
        a1 = [a, 0., 0.]
        a2 = [0.5*a, np.sqrt(3.)/2*a, 0.]
        a3 = [0., 0., c]
    else:
        raise ValueError('unsupported lattice: {}'.format(lattice_id))
    return a1,a2,a3

def reciprocal_lattice_vectors(lat1, lat2, lat3, crystallographic=True):
    """Compute the reciprocal lattice vectors.

    If not `crystallographic`, the computation includes
    the factor of 2*pi that is commmon in solid state physics
    """
    rlat1_xprod = np.cross(lat2,lat3)
    rlat2_xprod = np.cross(lat3,lat1)
    rlat3_xprod = np.cross(lat1,lat2)
    cellvol = np.dot(lat1,rlat1_xprod)
    rlat1 = rlat1_xprod/cellvol
    rlat2 = rlat2_xprod/cellvol
    rlat3 = rlat3_xprod/cellvol
    if not crystallographic:
        rlat1 *= 2*np.pi
        rlat2 *= 2*np.pi
        rlat3 *= 2*np.pi
    return rlat1, rlat2, rlat3

def symmetrize_points(all_hkl,rlat,space_group=None,symprec=1.E-6):
    # TODO: investigate whether or not the symmetrization
    # can just make use of the point group symmetries...
    reduced_hkl = copy.deepcopy(all_hkl)
    n_pts = all_hkl.shape[0]
    hkl_mults = np.ones(n_pts,dtype=int)
    lat_pts = np.dot(all_hkl,rlat.T)
    # rank the hkl points uniquely: higher rank means more likely to keep the point
    hkl_range = np.max(all_hkl,axis=0)-np.min(all_hkl,axis=0)
    hkl_rank = all_hkl[:,0]*(hkl_range[1]+1)*(hkl_range[2]+1) + all_hkl[:,1]*(hkl_range[2]+1) + all_hkl[:,2]
    sym_ops = []
    if space_group:
        if space_group in symmetries.symmetry_operations:
            sym_ops = symmetries.symmetry_operations[space_group]
    for op in sym_ops:
        sym_pts = np.dot(op,lat_pts.T).T 
        # get difference matrix between lat_pts and sym_pts.
        # lat_pts and sym_pts each have shape (N_points,3).
        # to broadcast subtraction of sym_pts across all lat_pts,
        # give lat_pts an extra dimension, transpose sym_pts, and subtract.
        # lat_pts[:,:,newaxis].shape = (N_points,3,1)
        # sym_pts.T.shape = 3,N_points
        # (lat_pts[:,:,newaxis]-sym_pts.T).shape = (N_points,3,N_points)
        lat_sym_diffs = lat_pts[:,:,np.newaxis] - sym_pts.T
        # get the scalar distances by taking vector norms along axis 1
        lat_sym_dists = np.linalg.norm(lat_sym_diffs,axis=1)
        # for each sym_pt, find the nearest lat_pt and the corresponding distance
        hkl_idx = np.arange(reduced_hkl.shape[0])
        min_dist_idx = np.argmin(lat_sym_dists,axis=1)
        # TODO: retrieve the min_dist using min_dist_idx
        min_dist = np.min(lat_sym_dists,axis=1)
        # get the set of indices to drop 
        # (those that mapped to within symprec of another point),
        # and use hkl_rank to decide which point to keep
        mapped_hkl_rank = hkl_rank[min_dist_idx]
        idx_to_drop = (min_dist_idx != hkl_idx) & (min_dist < symprec) & (hkl_rank < mapped_hkl_rank) 

        if any(idx_to_drop):
            idx_to_keep = np.invert(idx_to_drop) 
            hkl_mults[min_dist_idx[idx_to_drop]] += hkl_mults[idx_to_drop]
            hkl_mults = hkl_mults[idx_to_keep] 
            reduced_hkl = reduced_hkl[idx_to_keep,:] 
            lat_pts = lat_pts[idx_to_keep,:] 
            hkl_rank = hkl_rank[idx_to_keep] 

    return reduced_hkl,hkl_mults

# define coordinates for the extra sites
# that are added to centered lattices
centering_coords = dict(
    P = np.array([
        [0.,0.,0.]]),
    C = np.array([
        [0.,0.,0.],
        [0.5,0.5,0.]]),
    I = np.array([
        [0.,0.,0.],
        [0.5,0.5,0.5]]),
    F = np.array([
        [0.,0.,0.],
        [0.5,0.5,0.],
        [0.5,0.,0.5],
        [0.,0.5,0.5]]),
    HCP = np.array([
        [0.,0.,0.],
        [2./3,1./3,0.5]]),
    #A = np.array([
    #    [0.,0.,0.],
    #    [0.,0.5,0.5]]),
    #B = np.array([
    #    [0.,0.,0.],
    #    [0.5,0,0.5]]),
    #R = np.array([
    #    [2./3,1./3,1./3],
    #    [1./3,2./3,2./3]]),
    #D = np.array([
    #    [1./3,1./3,1./3],
    #    [2./3,2./3,2./3]])
    )


