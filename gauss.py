import cvxopt
import numpy as np
import scipy as sp
import scipy.linalg

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