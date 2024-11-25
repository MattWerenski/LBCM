import cvxopt
import numpy as np
import scipy as sp
import scipy.linalg
import torch
from torch.autograd import Variable

sqrtm = sp.linalg.sqrtm
inv = sp.linalg.inv

def wass_dist(A,B,A_h=None):
    '''
    wass_dist - Computes the Wasserstein distance between N(0,A) and N(0,B)

    :param A:   (d x d) np array for the covariance of the first variable
    :param B:   (d x d) np array for the covariance of the second variable
    :param A_h: (d x d) np array containing a pre-computed square root of A
    :return:    Wasserstein distance between N(0,A) and N(0,B)
    '''
    if A_h is None:
        A_h = sqrtm(A)
    return np.sqrt(np.trace(A + B - 2 * sqrtm(A_h @ B @ A_h)))



def true_bc(S_arr, lam, iters=20, initial=None):
    '''
    true_bc - Computes the true barycenter of the matrices in S_arr with coordinate lam
              using the algorithm from Chewi et al.

    :param S_arr:   list of p (d x d) np arrays of covariance matrices
    :param lam:     (p) np array containing the coordinate
    :param iters:   number of iterations to run
    :param initial: starting point of the iterations, defaults to S_arr[0]
    :return:        (d x d) np array of the covariance matrix
    '''
    p = len(S_arr)
    
    if initial is None:
        S = S_arr[0]
    else:
        S = initial
    
    dim = S.shape[0]
    
    for i in range(iters):
        S_h = sqrtm(S)   # matrix square root
        S_nh = sqrtm(inv(S)) # inverse matrix square root
        
        # first line of Chewi alg 1, integration
        T = np.zeros((dim,dim))
        for j in range(p):
            T = T + lam[j] * sqrtm(S_h @ S_arr[j] @ S_h)
            
        # update covariance using second formula
        S = S_nh @ T @ T @ S_nh
    
    return S



def mccann_interp(A,B,eta=0.5):
    '''
    mccann_interp - Computes the McCann interpolation of two matrices

    :param A:   (d x d) np array covariance matrix of the source
    :param B:   (d x d) np array covariance matrix of the traget
    :param eta: float between 0 and 1 representing how far to interpolate
    :return:    (d x d) np array of the interpolation
    '''
    A_h = sqrtm(A) # half power
    A_nh = sqrtm(inv(A)) #neg half
    
    # optimal map from A to B
    opt_map = A_nh @ sqrtm(A_h @ B @ A_h) @ A_nh
    
    # map for the McCann interpolation going eta amount
    interp_map = (1 - eta) * np.eye(opt_map.shape[0]) + eta * opt_map
    
    # Formula for cov. under linear transform
    return interp_map @ A @ interp_map



def lbcm_bc(refs, base, lam, C_refs=None):
    '''
    lbcm_bc - Computes the linear bary center for the given refs, base, and coordinate

    :param refs:   p (d x d) np arrays for the reference covariances
    :param base:   (d x d) np array for the base covariance
    :param lam:    (p) np array of the coordinates
    :param C_refs: pre-computed C matrices (linear transforms of base to references)
    :return:       the linear bary center wrt the given parameters
    '''
    if C_refs is None:
        S_h = sqrtm(base)
        S_nh = sqrtm(inv(base))

        C_refs = np.array([S_nh @ sqrtm(S_h @ Si @ S_h) @ S_nh for Si in refs])
        
    avg_map = np.tensordot(lam, C_refs, 1)
    return avg_map @ base @ avg_map.T



def opt_lam_grad_norm(refs, new, initial=None):
    '''
    opt_lam_grad_norm - Finds the optimal lambda which minimizes the norm of the gradient
                        in the BCM objective (approximately solves the BCM objective)

    :param refs:    list of p (d x d) np arrays for the refernce covariance matrices
    :param new:     (d x d) np arrays for the covariance to approximate
    :param initial: (p) np array of the initialization for the optimization procedure
    :return:        (p) np array of the optimal coordinate
    '''
    p = len(refs)
    
    # constructs the PSD matrix and Linear vector
    
    # pre-computed for many formulas
    new_h = sqrtm(new)
    new_nh = sqrtm(inv(new))
    
    # M_i, L_i, and R_i matrics 
    M_arr = [sqrtm(new_h @ Si @ new_h) for Si in refs]
    L_arr = [new_nh @ Mi for Mi in M_arr]
    R_arr = [Mi @ new_nh for Mi in M_arr]
    
    D = np.array([np.einsum('ij,ji->',Li,Rj) for Li in L_arr for Rj in R_arr]).reshape((p,p))
    e = -2 * np.array([np.trace(Mi) for Mi in M_arr])
    z = np.trace(new)
    
    cvxopt.solvers.options['show_progress'] = False
    
    # see docs for more info https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # P,q - specify the objective
    # G,h - specify non-negative constraints
    # A,b - sum-to-one constraint
    
    A = cvxopt.matrix(np.ones((1,p))) # for equality constraint
    b = cvxopt.matrix(1.0, (1,1))
    G = cvxopt.matrix(-np.eye(p)) # for inequality constraints
    h = cvxopt.matrix(0.0, (p,1))
    
    if initial is None:
        init = cvxopt.matrix(1/p, (p,1))
    else:
        init = cvxopt.matrix(initial)
    
    # solves the optimization
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P=cvxopt.matrix(2*D), q=cvxopt.matrix(e), G=G, h=h, A=A,b=b, initvals={'x':init})
    
    solution['primal objective'] += z
    solution['dual objective'] += z
    return np.array(solution['x']).squeeze()



def opt_lam_lbcm(refs, base, new, initial=None):
    '''
    opt_lam_lbcm - finds the optimal lambda in the LBCM movie by solving a QP

    :param refs:    list of p (d x d) np arrays for the refernce covariance matrices
    :param base:    (d x d) np array for the base measure in the LBCM
    :param new:     (d x d) np arrays for the covariance to approximate
    :param initial: (p) np array of the initialization for the optimization procedure
    :return:        (p) np array of the optimal coordinate
    '''
    p = len(refs)
    
    # pre-computed for many formulas
    S_h = sqrtm(base)
    S_nh = sqrtm(inv(base))
    
    C_refs = [S_nh @ sqrtm(S_h @ Si @ S_h) @ S_nh for Si in refs]
    C_new = S_nh @ sqrtm(S_h @ new @ S_h) @ S_nh
    
    diffs = np.array([C_ref - C_new for C_ref in C_refs])
    
    #inner_prods = np.trace(np.einsum('ijk,ljk->iljk', diffs, diffs @ base), axis1=2, axis2=3)
    inner_prods = np.einsum('ijk,lkj->il', diffs, diffs @ base)
    
    cvxopt.solvers.options['show_progress'] = False
    
    # see docs for more info https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # P,q - specify the objective
    # G,h - specify non-negative constraints
    # A,b - sum-to-one constraint
    
    A = cvxopt.matrix(np.ones((1,p))) # for equality constraint
    b = cvxopt.matrix(1.0, (1,1))
    G = cvxopt.matrix(-np.eye(p)) # for inequality constraints
    h = cvxopt.matrix(0.0, (p,1))
    
    if initial is None:
        init = cvxopt.matrix(1/p, (p,1))
    else:
        init = cvxopt.matrix(initial)
    
    # solves the optimization
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(
        P=cvxopt.matrix(2*inner_prods), 
        q=cvxopt.matrix(0.0, (p,1)),
        G=G, 
        h=h, 
        A=A,
        b=b, 
        initvals={'x':init}
    )

    return np.array(solution['x']).squeeze(), np.array(C_refs)

def opt_lam_mle(refs, emp, gamma=0.0003, iters=500, sqrt_iter=10, fixed_iter=10):
    '''
    opt_lam_mle - finds the Maximum Likelihood Estimator lambda using gradient descent

    :param refs:       Reference measures as a list of p np.arrays of size [dim, dim]
    :param emp:        measure to approximate as an np array of size [dim, dim]
    :param gamma:      step size parameter
    :param iters:      number of iterations to run
    :param sqrt_iter:  number of iterations for computing the matrix square roots
    :param fixed_iter: number of fixed point iterations to run

    :return: optimal lambda found by the algorithm
    '''
    refs = torch.tensor(refs, dtype=torch.float32)
    emp = torch.tensor(emp, dtype=torch.float32)
    
    p = len(refs)
    lam = torch.tensor(np.ones(p)/p, dtype=torch.float32, requires_grad=True)
    
    # gradient descent iteration
    for i in range(iters):
        # construct the barycenter
        bc = barycenter_torch(refs, lam, emp, sqrt_iter=sqrt_iter, fixed_iter=fixed_iter)

        # compute the loss between the bc and empirical
        loss = kl_loss(emp, bc)

        # back prop
        loss.backward(retain_graph=True)

        with torch.no_grad():
            prev_lam = lam.clone()
            lam = lam - gamma * lam.grad # gradient descent
            lam = proj_vec(lam) # project back onto simplex
        lam.requires_grad = True
        
        if (prev_lam - lam).abs().sum() < 0.0000001:
            break

    return lam.detach().numpy()


def proj_vec(lam):
    '''
    proj_vec - projects the vector onto the probability simplex

    :param lam: tensor of size [dim] to be projected

    :return: the L2 projection of lam onto the simplex
    '''
    p = lam.shape[0] 
    dtype = lam.dtype
    
    mat0 = torch.zeros(p,p)
    mat0[torch.arange(p)>=torch.transpose(torch.arange(p).unsqueeze(0),1,0)] = 1
    mat0 = mat0.to(dtype=dtype)
    
    mat1 = torch.diag(1/(torch.arange(p)+1))
    mat1 = mat1.to(dtype=dtype)
    
    [U,_] = torch.sort(lam,descending=True)
    U_ = U + (1 - U @ mat0) @ mat1
    rho = torch.max((U_ > 0).nonzero())

    U[torch.arange(p) > rho] = 0
    rho = (1 - U.sum()) / (rho + 1)

    lam = lam + rho
    lam = torch.max(lam, torch.tensor([0.0],dtype=dtype))
    return lam

def kl_loss(S_0, S_1):
    '''
    kl_loss - computes the important parts of the KL divergence between S_0 and S_1 for
    optimizing in S_1

    :param S_0: tensor of size [dim, dim] representing the first measure
    :param S_1: tensor of size [dim, dim] representing the second measure

    :return: D_KL( N(0,S_0) || N(0,S_1) ), but only including the terms involving S_1-
    '''
    inv = torch.inverse(S_1)
    # slogdet returns a tuple containing (sign, logabsdet)
    return torch.trace(inv @ S_0) + torch.slogdet(S_1)[1] 

def barycenter_torch(refs, lam, init, sqrt_iter=10, fixed_iter=10):
    '''
    barycenter_torch - runs the fixed-point iteration algorithm to find 
    the barycenter of the given reference measures with cooridinate lambda, 
    starting from the given initial point

    :param refs:       tensor of size [p, dim, dim] with refs[i] being the i'th ref measure
    :param lam:        barycentric coordinate
    :param sqrt_iter:  number of iterations to run the square root calculation for
    :param fixed_iter: numbder of fixed-point iterations

    :return: tensor of size [dim, dim] containing the barycenter
    '''
    [p, dim, _] = refs.shape
    
    # use the initial guess provided spread p times (size [p, dim, dim])
    bc = init.clone()
    for t in range(fixed_iter): 
        # compute the terms inside the integral / sum (size [p, dim, dim])
        #             expand because this function expects 3D tensors
        bc_sqrt = sqrt_newton_schulz_autograd(bc.view(1,dim,dim), sqrt_iter)[0]
        sum_terms = bc_sqrt @ refs @ bc_sqrt
        sqrt_terms = sqrt_newton_schulz_autograd(sum_terms, sqrt_iter)
        
        # summing along the first axis, adds the matrices together (size [dim,dim])
        S_t = (lam.view(p,1,1) * sqrt_terms).sum(0) 
        
        # compute the -1/2 power of the current bc (size [dim,dim])
        # we take the 0'th because bc_sqrt was repeating the p times on first axis.
        bc_sqrt_inv = torch.inverse(bc_sqrt)
        
        bc = bc_sqrt_inv @ S_t @ S_t @ bc_sqrt_inv
    return bc


def sqrt_newton_schulz_autograd(mats, numIters=10):
    '''
    sqrt_newron_schulz_autograd - runs the Newton-Shulz algorithm for 
    finding the square root of a matrix and does so in a differentiable manner

    :param mats:     tensor of size [p, dim, dim] where we take the sqrt the matrices
                     stored as mats[i] for i = 0,...,p-1
    :param numIters: number of iterations to run for

    :return: tensor os size [p,dim,dim] where sqrt_mats[i] is the sqrt of mats[i]
    '''
    dtype = mats.dtype
    p = mats.data.shape[0]
    dim = mats.data.shape[1]
    
    mat_norms = (mats @ mats).sum(dim=(1,2)).sqrt()
    
    # divide each matrix by its norm (size [p, dim, dim])
    Y = mats.div(mat_norms.view(p, 1, 1).expand_as(mats))
    
    # repeated identy matrix (size [p, dim, dim])
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(p, 1, 1).type(dtype), requires_grad=False)
    
    # repeated identy matrix (size [p, dim, dim])
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(p, 1, 1).type(dtype), requires_grad=False)

    # Newton-Schulz iterations
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y)) # bmm multiplies Z[i] @ Y[i]
        Y = Y.bmm(T)
        Z = T.bmm(Z)
        
    sqrt_mats = Y * torch.sqrt(mat_norms).view(p, 1, 1).expand_as(mats)
    
    return sqrt_mats

