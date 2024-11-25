import cvxopt
import numpy as np
import ot

def coupling_to_map(coupling, target_supp):
    '''
    coupling_to_map - Given a coupling and the target support returns the entropic map
                      evaluated at the source points
    
    :param coupling:    (m x n) np array correspond to a coupling
    :param target_supp: (n x d) np array corresponding to the support of the target
    :return:            (m x d) np array with the i'th row being the image of the i'th 
                        point in the source
    '''
    normalized_rows = coupling / (coupling.sum(1)[:,None])
    return normalized_rows @ target_supp



def compute_map(source, target, reg=0.001):
    '''
    compute_map - Computes the entropic map from the base image to each reference image

    :param source: [mass, support] pair representing the source measure with shapes [(m), (m x d)]
    :param target: [mass, support] pair representing the target measure with shapes [(n), (n x d)]
    :param reg:    entropic regularization parameter
    :return:       (m x d) np arrays corresponding to the entropic maps from the source to target
    '''
    
    [source_mass, source_supp] = source
    [target_mass, target_supp] = target
    
    distance = ot.utils.dist(source_supp, target_supp, metric='sqeuclidean') / 2
    if reg == 0:
        coupling = ot.lp.emd(source_mass, target_mass, distance)
    else:
        coupling = ot.sinkhorn(source_mass, target_mass, distance, reg)
    return coupling_to_map(coupling, target_supp)



def compute_inner_products(base, references, new_measure, reg=0.001, ref_maps=[]):
    '''
    compute_inner_products - Computes the inner products used in the quadratic form

    :param base_image:  [mass, support] pair representing the base measure in the LBCM
    :param references:  list of p [mass, support] pairs representing the reference measures 
                        in the LBCM
    :param new_measure: [mass, support] pairs representing the new measure to find the 
                        LBCM coordinates of
    :param reg:         regularization parameter in the entropic map
    :param ref_maps:    if provided, uses these maps instead of re-computing them.
                        Pass [] or leave empty to compute maps.
    :return:            The computed inner products as a (p x p) np array, the entropic map
                        from the base to the new measure, and the entropic map from the base
                        to the references
    '''
    # if you don't pass the reference maps it computes them here (may be slow)
    if len(ref_maps) == 0:
        ref_maps = [compute_map(base, ref, reg=reg) for ref in references]
        new_map = compute_map(base, new_measure, reg=reg)
    
    else:
        new_map = compute_map(base, new_measure, reg=reg)
    
    diffs = [ref_map - new_map for ref_map in ref_maps]
    
    base_mass = base[0]
    # fancy way of computing the inner products in one line
    inner_products = np.einsum('ikd,jkd,k->ij', diffs, diffs, base_mass)
    
    return inner_products, new_map, ref_maps



def find_coordinate(base, references, new_measure, reg=0.001, ref_maps=[]):
    '''
    find_coordinate - Finds the optimal coordinate in the LBCM

    :param base_image:  [mass, support] pair representing the base measure in the LBCM
    :param references:  list of p [mass, support] pairs representing the reference measures 
                        in the LBCM
    :param new_measure: [mass, support] pairs representing the new measure to find the 
                        LBCM coordinates of
    :param reg:         regularization parameter in the entropic map
    :param ref_maps:    if provided, uses these maps instead of re-computing them.
                        Pass [] or leave empty to compute maps
    :return:            (p) np array corresponding to the estimated coordinate (p x p) np array 
                        for the inner prodcuts, the entropic map for the new image, and the maps
                        from the base to the references.
    '''
    p = len(references)
    inner_products, new_map, ref_maps = compute_inner_products(base, references, 
                                                      new_measure, reg=reg, ref_maps=ref_maps)
        
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
        P=cvxopt.matrix(inner_products), 
        q=cvxopt.matrix(0.0, (p,1)),
        G=G, 
        h=h, 
        A=A,
        b=b, 
        initvals={'x':init}
    )

    return np.array(solution['x']).squeeze(), inner_products, new_map, ref_maps



def synthesize(base, references, lam, reg=0.001, ref_maps=[]):
    '''
    synthesize - Computes the measure corresponding to the given coordinate (lam) for the LBCM
                 with given base and references

    :param base: [mass, support] pair representing the base measure in the LBCM
    :param references: list of p [mass, support] pairs representing the reference measures 
                       in the LBCM
    :param lam:        (p) np array of the coordinate to synthesize
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] or leave empty to compute maps
    :return:           The [mass, support] pair for the synthesized measure
    '''

    if len(ref_maps) == 0:
        # if needed, computes the maps to the references
        ref_maps = [compute_map(base, ref, reg=reg) for ref in references]
    
    # the synthesized measure has the same masses as the base
    base_mass = base[0]
    # fancy one-liner to average the maps together and find the support
    synth_support = np.einsum('i,ijk->jk', lam, ref_maps)
    
    return [base_mass, synth_support]

def particle_synthesis(references, lam, initial, iterations=200, step_size=0.05, reg=0.0):
    '''
    particle_synthesis - Particle method for estimating the barycenter of a set of measures
    
    :param references: list of [mass, support] pairs representing the references
    :param lam:        barycentric coordinate to estimate the barycenter of
    :param initial:    starting measure for the fixed point process
    :param iterations: number of fixed point iterations to run
    :param step_size:  how far to move at each iteration. Should be in (0,1)
    :param reg:        regularization parameter in the entropic map
    :return:           The [mass, support] pair for the synthesized measure
    '''

    synth_measure = initial
    for i in range(iterations):
        # update all the maps
        curr_maps = np.array([compute_map(synth_measure, ref, reg=reg) for ref in references])
        
        # compute the weighted average
        averaged_maps = (lam[:,None,None] * curr_maps).sum(0)

        # update the support
        synth_measure[1] = (1-step_size) * synth_measure[1] + step_size * averaged_maps
        
    return synth_measure

def project(base, references, new_measure, reg=0.001, ref_maps=[]):
    '''
    project - Estimates the coordinate lambda and uses it to create the linear barycenter
              which is closest to the given measure

    :param base_image:  [mass, support] pair representing the base measure in the LBCM
    :param references:  list of p [mass, support] pairs representing the reference measures 
                        in the LBCM
    :param new_measure: [mass, support] pairs representing the new measure to find the 
                        LBCM coordinates of
    :param reg:         regularization parameter in the entropic map
    :param ref_maps:    if provided, uses these maps instead of re-computing them.
                        Pass [] or leave empty to compute maps
    :return:            the [mass, support] pair for the projection onto the LBCM with given base and
                        references, the coordinate to which it corresponse, the computed inner products,
                        and the entropic map from the base to the references
    '''

    if len(ref_maps) == 0:
        # if needed, computes the maps to the references
        ref_maps = [compute_map(base, ref, reg=reg) for ref in references]
    
    # first find the coordinate
    lam, inner_products, new_map, _ = find_coordinate(
        base, references, new_measure, reg=reg, ref_maps=ref_maps)
    
    # then synthesize using that coordinate
    projected_measure = synthesize(base, references, lam, reg=reg, ref_maps=ref_maps)
    
    return projected_measure, lam, inner_products, ref_maps


