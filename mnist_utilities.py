'''
Functions for working with MNIST digits
'''

import cvxopt
import numpy as np
import ot
import ot.bregman

DIM = 28

#======== HELPER FUNCTIONS ========

def plan_to_map(plan, support):
    marginal = plan.sum(1)
    has_mass = marginal > 0
    no_mass = 1 - has_mass
    
    # if the rows of plan summed to 1, this would be the average location
    # that the plan sends each coordinate to 
    unweighted = plan @ support 
    
    
    # plan.shape[0] is number of points in the source
    # support.shape[1] is the dimension we work in
    weighted = np.zeros((plan.shape[0],support.shape[1]))
    
    # here we account for the fact that the rows don't sum to 1
    weighted[has_mass,:] = unweighted[has_mass,:] / marginal[has_mass,np.newaxis]
    
    # if theres no mass there, then this value doesn't matter
    # so we set it to map to itself
    weighted[no_mass,:] = support[no_mass,:]
    
    return weighted

def crop(images):
    rows = [img.sum(1) for img in images]
    cols = [img.sum(0) for img in images]
    
    rows = np.asarray(rows).sum(0)
    cols = np.asarray(cols).sum(0)
    
    top = 0
    while rows[top] == 0:
        top += 1
    bottom = DIM - 1
    while rows[bottom] == 0:
        bottom -= 1
    bottom += 1
    left = 0
    while cols[left] == 0:
        left += 1
    right = DIM - 1
    while cols[right] == 0:
        right -= 1
    right += 1
        
    cropped = [img[top:bottom, left:right] for img in images]
    bounds = (top, bottom, left, right)
    
    return cropped, bounds
    
def uncrop(image, bounds):
    uncropped = np.zeros((DIM, DIM))
    uncropped[bounds[0]:bounds[1], bounds[2]:bounds[3]] = image
    return uncropped
    
'''
mnist_utilities.convolutional_barycenter
    Computes the Wasserstein barycenter of the reference measures for the
    given weights. This function relies on the POT library 
    NOTE: This function is much better than the one above 
    and should be preferred in general.
    
parameters
    refs - list of reference measures, each represented
        as a 2D array of intensities, normalized to sum to 1.
    weight - mixture weight for which the barycenter is computed
    
returns
    the barycenter of the given references using the mixture weight
    as a DIM by DIM np.array.
    
'''
def convolutional_barycenter(refs, weights, threshold=0.00001, entropy=0.003):
    # don't use tiny weights
    weights_used = weights[weights > threshold]
    weights_used = weights_used / weights_used.sum()
    
    refs_used = refs[weights > threshold,:,:]
    
    cropped_refs, bounds = crop(refs_used)
    cropped_refs = np.array(cropped_refs)
    
    cropped_bc = ot.bregman.convolutional_barycenter2d(cropped_refs, entropy, weights=weights_used,
                                                       numItermax=50000,stopThr=1e-7)
    
    return uncrop(cropped_bc, bounds)
    

'''
mnist_utilities.inner_products
    Computes the approximate inner products of the maps from base to ref
    in the tangent space at the base and returns a matrix of these.
    
parameters
    base - the measure we are trying to approximate with a barycenter
    refs - list of reference measures, each represented
        as a 2D array of intensities, normalized to sum to 1.
    mask - hides portions of the reference images to match corruption in base
    
returns
    a p by p numpy array A with A_ij the inner product for references i and j
'''

def inner_products(base, refs, supp=None):
    
    [dim1, dim2] = base.shape
    
    if supp is None: # faster to cache than re-compute
        supp = []
        for i in range(dim1):
            for j in range(dim2):
                supp += [[i,j]]
        supp = np.array(supp)
        
    # flattens images to arrays
    base_uw = base.reshape(dim1*dim2)
    ref_uws = [ref.reshape(dim1*dim2) for ref in refs]
    
    # take out zero mass parts of the base
    base_used = base_uw > 0
    base_dist = base_uw[base_used]
    base_dist = base_dist / base_dist.sum()
    base_supp = supp[base_used, :]
    
    opt_maps = []
    
    for ref_uw in ref_uws:
        
        # take out zero mass parts of the reference
        ref_used = ref_uw > 0
        ref_dist = ref_uw[ref_used]
        ref_dist = ref_dist / ref_dist.sum()
        ref_supp = supp[ref_used,:]
        
        M = ot.dist(base_supp, x2=ref_supp, metric='sqeuclidean')
        
        gamma = ot.lp.emd(base_dist, ref_dist, M)
        
        opt_maps += [plan_to_map(gamma, ref_supp)]
        
    # pre-perform the subtraction of the identity
    adjusted_maps = [opt_map - base_supp for opt_map in opt_maps]
    
    p = refs.shape[0]
    A = np.zeros((p,p))
    
    # actually fill in the A matrix
    for i, map_i in enumerate(adjusted_maps):
        for j, map_j in enumerate(adjusted_maps):
            ip = np.dot((map_i * map_j).sum(1), base_dist)
            A[i,j] = ip
            
    return A
    
'''
opt_utilities.solve
    Actually solves the minimization procedure we've defined, given the 
    evaluation of the inner products in the tangent space.
    
parameters
    inner_products - p by p matrix of inner products in the tangent space.

returns
    the optimal mixture weight
'''
def solve(inner_products, return_val=False):
    p = inner_products.shape[0]
    
    P = cvxopt.matrix(inner_products)
    q = cvxopt.matrix(np.zeros(p))
    G = cvxopt.matrix(-np.eye(p))
    h = cvxopt.matrix(np.zeros(p))
    A = cvxopt.matrix(np.ones((1,p)))
    b = cvxopt.matrix(np.ones((1,1)))

    cvxopt.solvers.options['show_progress'] = False
    
    soln = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    lam = np.squeeze(np.array(soln['x']))
    
    if return_val:
        return [lam, soln['primal objective']]
    return lam
    