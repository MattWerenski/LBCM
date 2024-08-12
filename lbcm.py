import cvxopt
import numpy as np
import ot

from images import image_to_empirical

def coupling_to_map(coupling, target_support):
    '''
    coupling_to_map - Given a coupling and the target support returns the entropic map
                      evaluated at the source points
    
    :param coupling:       (m x n) np array correspond to a coupling
    :param target_support: (n x 2) np array corresponding to the support of the target
    :return:               (m x 2) np array with the i'th row being the image of the i'th point in the source
    '''
    normalized_rows = coupling / (coupling.sum(1)[:,None])
    return normalized_rows @ target_support

def compute_maps(base_image:np.array, ref_images, reg=0.001):
    '''
    compute_maps - Computes the entropic map from the base image to each reference image

    :param base_image: (m x n) np array corresponding to the image (not as an empirical measure)
    :param ref_images: list of np arrays corresponding to the reference images
    :param reg:        regularization parameter for the entropic map
    :return:           list of (m x 2) np arrays corresponding to the entropic maps from the base to the refs
    '''

    base_support, base_mass = image_to_empirical(base_image)
    maps = []
    for ref_image in ref_images:
        ref_support, ref_mass = image_to_empirical(ref_image)
        dist = ot.utils.dist(base_support, ref_support, metric='sqeuclidean') / 2
        if reg == 0:
            coupling = ot.lp.emd(base_mass, ref_mass, dist)
        else:
            coupling = ot.sinkhorn(base_mass, ref_mass, dist, reg)
        maps += [coupling_to_map(coupling, ref_support)]
    return maps

def compute_inner_products(base_image, ref_images, new_image, reg=0.001, ref_maps=[]):
    '''
    compute_inner_products - Computes the inner products used in the quadratic form

    :param base_image: base image in the LBCM
    :param ref_images: list of p reference images in the LBCM
    :param new_image:  new image to find the LBCM coordinates of
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] to compute maps
    :return:           (p x p) array for the inner prodcuts, the entropic map for the new image,
                       and the maps from the base to the references.
    '''
    # if you don't pass the refs it computes them here (may be slow)
    if len(ref_maps) == 0:
        maps = compute_maps(base_image, ref_images + [new_image], reg=reg)
        ref_maps = maps[0:-1]
        new_map = maps[-1]
    
    else:
        new_map = compute_maps(base_image, [new_image], reg=reg)[0]
    
    diffs = [ref_map - new_map for ref_map in ref_maps]
    _, base_mass = image_to_empirical(base_image)
    
    inner_products = np.einsum('ikd,jkd,k->ij', diffs, diffs, base_mass)
    
    return inner_products, new_map, ref_maps

def find_coordinate_lbcm(base_image, ref_images, new_image, reg=0.001, ref_maps=[]):
    '''
    find_coordinate_lbcm - Finds the optimal coordinate in the LBCM

    :param base_image: base image in the LBCM
    :param ref_images: list of p reference images in the LBCM
    :param new_image:  new image to find the LBCM coordinates of
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] to compute maps
    :return:           p np array corresponding to the estimated coordinate (p x p) np array 
                       for the inner prodcuts, the entropic map for the new image, and the maps 
                       from the base to the references.
    '''
    p = len(ref_images)
    inner_products, new_map, ref_maps = compute_inner_products(base_image, ref_images, 
                                                      new_image, reg=reg, ref_maps=ref_maps)
        
    # see docs for more info https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # P,q - specify the objective
    # G,h - specify non-negative constraints
    # A,b - sum-to-one constraint
    
    A = cvxopt.matrix(np.ones((1,p))) # for equality constraint
    b = cvxopt.matrix(1.0, (1,1))
    G = cvxopt.matrix(-np.eye(p)) # for inequality constraints
    h = cvxopt.matrix(0.0, (p,1))
    
    init = cvxopt.matrix(1/p, (p,1))
    
    # solves the optimization
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(
        P=cvxopt.matrix(2*inner_products), 
        q=cvxopt.matrix(0.0, (p,1)),
        G=G, 
        h=h, 
        A=A,
        b=b, 
        initvals={'x':init}
    )

    return np.array(solution['x']).squeeze(), inner_products, new_map, ref_maps

def synthesize_lbcm(base_image, ref_images, new_image, reg=0.001, lam=None, ref_maps=[]):
    '''
    synthesize_lbcm - Estimates the coordinate lambda and uses it to create the linear barycenter

    :param base_image: base image in the LBCM
    :param ref_images: list of p reference images in the LBCM
    :param new_image:  new image to find the LBCM coordinates of
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] to compute maps
    :return:           empirical linear barycenter as a (l,2) np array for the supoprt and 
                       (l) np array for the mass p np array corresponding to the estimated coordinate 
                       (p x p) np array for the inner prodcuts, the entropic map for the new image, 
                       and the maps from the base to the references.
    '''

    if lam is None:
        lam, _, _, ref_maps = find_coordinate_lbcm(
            base_image, ref_images, new_image, reg=reg, ref_maps=ref_maps)
    
    _, synth_mass = image_to_empirical(base_image)
    synth_support = np.einsum('i,ijk->jk', lam, ref_maps)
    
    return synth_support, synth_mass
