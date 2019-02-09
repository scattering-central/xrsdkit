from collections import OrderedDict
import copy

import numpy as np

from .. import definitions as xrsdefs

# define all symmetry operations: Mx=x'
# the inversion operator
inversion = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
# three axial mirrors 
mirror_x = np.array([[-1,0,0],[0,1,0],[0,0,1]])
mirror_y = np.array([[1,0,0],[0,-1,0],[0,0,1]])
mirror_z = np.array([[1,0,0],[0,1,0],[0,0,-1]])
# six diagonal mirrors 
mirror_x_y = np.array([[0,-1,0],[-1,0,0],[0,0,1]])
mirror_y_z = np.array([[1,0,0],[0,0,-1],[0,-1,0]])
mirror_z_x = np.array([[0,0,-1],[0,1,0],[-1,0,0]])
mirror_nx_y = np.array([[0,1,0],[1,0,0],[0,0,1]])
mirror_ny_z = np.array([[1,0,0],[0,0,1],[0,1,0]])
mirror_nz_x = np.array([[0,0,1],[0,1,0],[1,0,0]])

# enumerate valid symmetry operations for each point group:
# note that the symmetrization algorithm retains points
# with higher h values, and then (for equal h), higher k values,
# and then (for equal h and k), higher l values.
symmetry_operations = OrderedDict.fromkeys(xrsdefs.all_point_groups)

# TODO: tabulate all symmetry operations
# that can be used to reduce the reciprocal space summation
# for a given space group.
# TODO: determine whether or not this can be done based solely on the point group
# associated with the space group.
symmetry_operations['P1'] = [] 
symmetry_operations['P-1'] = [inversion] 
symmetry_operations['Fm-3m'] = [\
    mirror_x,mirror_y,mirror_z,\
    mirror_x_y,mirror_y_z,mirror_z_x,\
    mirror_nx_y,mirror_ny_z,mirror_nz_x\
    # TODO: add the 3-fold x+y+z-rotoinversion
    ]
symmetry_operations['P6(3)/mmc'] = [\
    #mirror_z, mirror_x_y, mirror_nx_y
    ]

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
        if space_group in symmetry_operations:
            sym_ops = symmetry_operations[space_group]
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



