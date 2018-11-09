import copy

import numpy as np

from . import symmetries
from . import space_groups as sgs

def get_lattice_vectors(lattice_id,a=None,b=None,c=None,alpha=None,beta=None,gamma=None):
    # TODO: expand to support all lattices 
    if lattice_id in ['fcc']:
        a1 = [a, 0., 0.]
        a2 = [0., a, 0.]
        a3 = [0., 0., a]
    elif lattice_id in ['hcp']:
        a1 = [a, 0., 0.]
        a2 = [0.5*a, np.sqrt(3.)/2*a, 0.]
        a3 = [0., 0., np.sqrt(8.)/3.*a]
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

def symmetrize_points(all_hkl,rlat,point_group,symprec=1.E-6):
    reduced_hkl = copy.deepcopy(all_hkl)
    n_pts = all_hkl.shape[0]
    hkl_mults = np.ones(n_pts,dtype=int)
    lat_pts = np.dot(all_hkl,rlat.T)
    # rank the hkl points uniquely: higher rank means more likely to keep the point
    hkl_range = np.max(all_hkl,axis=0)-np.min(all_hkl,axis=0)
    hkl_rank = all_hkl[:,0]*(hkl_range[1]+1)*(hkl_range[2]+1) + all_hkl[:,1]*(hkl_range[2]+1) + all_hkl[:,2]
    for op in symmetries.symmetry_operations[point_group]:
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

# map common lattice names to their Bravais lattice systems
lattice_map = {\
    'fcc':'cubic',\
    'bcc':'cubic',\
    'sc':'cubic',\
    'hcp':'hexagonal',\
    'hexagonal':'hexagonal',\
    'rhombohedral':'rhombohedral',\
    'tetragonal':'tetragonal',\
    'tetragonal-I':'tetragonal',\
    'orthorhombic':'orthorhombic',\
    'orthorhombic-I':'orthorhombic',\
    'orthorhombic-F':'orthorhombic',\
    'orthorhombic-C':'orthorhombic',\
    'monoclinic':'monoclinic',\
    'monoclinic-C':'monoclinic',\
    'triclinic':'triclinic'\
    }
    #'hexagonal-R':'hexagonal',\
    #'rhombohedral-D':'rhombohedral',\
    #'orthorhombic-A':'orthorhombic',\
    #'orthorhombic-B':'orthorhombic',\
    #'monoclinic-A':'monoclinic',\
    #'monoclinic-B':'monoclinic',\

# map common lattice names to centering specifiers
centering_map = {\
    'fcc':'F',\
    'bcc':'I',\
    'sc':'P',\
    'hcp':'HCP',\
    'hexagonal':'P',\
    'rhombohedral':'P',\
    'tetragonal':'P',\
    'tetragonal-I':'I',\
    'orthorhombic':'P',\
    'orthorhombic-I':'I',\
    'orthorhombic-F':'F',\
    'orthorhombic-C':'C',\
    'monoclinic':'P',\
    'monoclinic-C':'C',\
    'triclinic':'P'\
    }
    #'hexagonal-R':'R',\
    #'rhombohedral-D':'D',\
    #'orthorhombic-A':'A',\
    #'orthorhombic-B':'B',\
    #'monoclinic-A':'A',\
    #'monoclinic-B':'B',\

default_high_sym_space_groups = {\
    'fcc':sgs.lattice_space_groups['cubic'][225],\
    'bcc':sgs.lattice_space_groups['cubic'][229],\
    'sc':sgs.lattice_space_groups['cubic'][221],\
    'hcp':sgs.lattice_space_groups['hexagonal'][194],\
    'hexagonal':sgs.lattice_space_groups['hexagonal'][194],\
    'rhombohedral':sgs.lattice_space_groups['rhombohedral'][166],\
    'tetragonal':sgs.lattice_space_groups['tetragonal'][123],\
    'tetragonal-I':sgs.lattice_space_groups['tetragonal'][139],\
    'orthorhombic':sgs.lattice_space_groups['orthorhombic'][47],\
    'orthorhombic-C':sgs.lattice_space_groups['orthorhombic'][65],\
    'orthorhombic-I':sgs.lattice_space_groups['orthorhombic'][71],\
    'orthorhombic-F':sgs.lattice_space_groups['orthorhombic'][69],\
    'monoclinic':sgs.lattice_space_groups['monoclinic'][10],\
    'monoclinic-C':sgs.lattice_space_groups['monoclinic'][12],\
    'triclinic':sgs.lattice_space_groups['triclinic'][2]\
    }
    #'hexagonal-R':,\
    #'rhombohedral-D':,\
    #'orthorhombic-A':,\
    #'orthorhombic-B':,\
    #'monoclinic-A':,\
    #'monoclinic-B':,\

default_low_sym_space_groups = {\
    'fcc':sgs.lattice_space_groups['cubic'][196],\
    'bcc':sgs.lattice_space_groups['cubic'][197],\
    'sc':sgs.lattice_space_groups['cubic'][195],\
    'hcp':sgs.lattice_space_groups['hexagonal'][168],\
    'hexagonal':sgs.lattice_space_groups['hexagonal'][168],\
    'rhombohedral':sgs.lattice_space_groups['rhombohedral'][146],\
    'tetragonal':sgs.lattice_space_groups['tetragonal'][75],\
    'tetragonal-I':sgs.lattice_space_groups['tetragonal'][79],\
    'orthorhombic':sgs.lattice_space_groups['orthorhombic'][16],\
    'orthorhombic-C':sgs.lattice_space_groups['orthorhombic'][21],\
    'orthorhombic-I':sgs.lattice_space_groups['orthorhombic'][23],\
    'orthorhombic-F':sgs.lattice_space_groups['orthorhombic'][22],\
    'monoclinic':sgs.lattice_space_groups['monoclinic'][3],\
    'monoclinic-C':sgs.lattice_space_groups['monoclinic'][5],\
    'triclinic':sgs.lattice_space_groups['triclinic'][1]\
    }
    #'hexagonal-R':,\
    #'rhombohedral-D':,\
    #'orthorhombic-A':,\
    #'orthorhombic-B':,\
    #'monoclinic-A':,\
    #'monoclinic-B':,\


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


