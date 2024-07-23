import cvxopt
import numpy as np
import scipy as sp
import scipy.linalg

sqrtm = sp.linalg.sqrtm
inv = sp.linalg.inv

# computes the wasserstein distance between N(0,A) and N(0,B)
def wass_dist(A,B,A_h=None):
    if A_h is None:
        A_h = sqrtm(A)
    return np.sqrt(np.trace(A + B - 2 * sqrtm(A_h @ B @ A_h)))

# computes the true barycenter using the algorithm of Chewi et al.
def true_bc(S_arr, lam, iters=20, initial=None):
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

# computs the eta inerpolation point from N(0,A) to N(0,B)
def mccann_interp(A,B,eta=0.5):
    A_h = sqrtm(A) # half power
    A_nh = sqrtm(inv(A)) #neg half
    
    # optimal map from A to B
    opt_map = A_nh @ sqrtm(A_h @ B @ A_h) @ A_nh
    
    # map for the McCann interpolation going eta amount
    interp_map = (1 - eta) * np.eye(opt_map.shape[0]) + eta * opt_map
    
    # Formula for cov. under linear transform
    return interp_map @ A @ interp_map

# computes the norm of the gradient at a given matrix
def grad_norm(S_arr, lam, S):
    p = len(S_arr)
    
    # constructs the PSD matrix and Linear vector
    
    # pre-computed for below
    S_h = sqrtm(S)
    S_nh = inv(sqrtm(S))
    
    # M_i, L_i, and R_i matrics 
    M_arr = [sqrtm(S_h @ Si @ S_h) for Si in S_arr]
    L_arr = [S_nh @ Mi for Mi in M_arr]
    R_arr = [Mi @ S_nh for Mi in M_arr]
    
    D = np.array([np.einsum('ij,ji->',Li,Rj) for Li in L_arr for Rj in R_arr]).reshape((p,p))
    e = -2 * np.array([np.trace(Mi) for Mi in M_arr])
    z = np.trace(S)
    
    return np.sqrt(lam.T @ D @ lam + lam @ e + z)

def lbcm_bc(refs, base, lam, C_refs=None):
    
    if C_refs is None:
        S_h = sqrtm(base)
        S_nh = sqrtm(inv(base))

        C_refs = np.array([S_nh @ sqrtm(S_h @ Si @ S_h) @ S_nh for Si in refs])
        
    avg_map = np.tensordot(lam, C_refs, 1)
    return avg_map @ base @ avg_map.T

# Finds the optimal lambda by minimizing the norm of the gradient
# this can be seen as minimizing an upper bound in the wass. distance

def opt_lam_grad_norm(S_arr, S, initial=None):
    p = len(S_arr)
    
    # constructs the PSD matrix and Linear vector
    
    # pre-computed for many formulas
    S_h = sqrtm(S)
    S_nh = sqrtm(inv(S))
    
    # M_i, L_i, and R_i matrics 
    M_arr = [sqrtm(S_h @ Si @ S_h) for Si in S_arr]
    L_arr = [S_nh @ Mi for Mi in M_arr]
    R_arr = [Mi @ S_nh for Mi in M_arr]
    
    D = np.array([np.einsum('ij,ji->',Li,Rj) for Li in L_arr for Rj in R_arr]).reshape((p,p))
    e = -2 * np.array([np.trace(Mi) for Mi in M_arr])
    z = np.trace(S)
    
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

# Finds the optimal lambda by minimizing the norm of the gradient
# this can be seen as minimizing an upper bound in the wass. distance

def opt_lam_lbcm(refs, base, new, initial=None):
    
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