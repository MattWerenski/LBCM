import numpy as np
import scipy as sp
import scipy.stats

from gauss import *

dim = 10
p = 10
nsamples = [64, 128, 256, 512]
trials = 2
results = {}

for n in nsamples:
    nresults = {
        'emp': np.zeros(trials),
        'grad_norm': np.zeros(trials),
        'lbcm_id': np.zeros(trials),
        'lbcm_bc': np.zeros(trials)
    }
    
    for t in range(trials):

        # generates a set of random PSD matrices with the same eigenvalues
        ortho_basis = sp.stats.special_ortho_group.rvs(dim)
        covs = []
        for _ in range(p):
            eigenvalues = np.abs(np.random.randn(dim))
            covs += [ortho_basis @ np.diag(eigenvalues) @ ortho_basis.T]
        covs = np.array(covs)

        # generates a random coordinate
        lam = np.random.dirichlet(np.ones(p))

        # computes the ground truth covariance matrix
        true_cov = true_bc(covs, lam)

        # and samples from it to construct the empirical matrix
        samples = sp.stats.multivariate_normal(np.zeros(dim),true_cov).rvs(n)
        emp_cov = samples.T @ samples / n

        nresults['emp'][t] = wass_dist(emp_cov, true_cov)

        # minimizes the norm of the gradient
        lam_grad_norm = np.array(opt_lam_grad_norm(covs, emp_cov)).squeeze()
        cov_grad_norm = true_bc(covs, lam_grad_norm)
        nresults['grad_norm'][t] = wass_dist(cov_grad_norm, true_cov)

        # LBCM with identity base
        lam_lbcm_id, C_refs = opt_lam_lbcm(covs, np.eye(dim), emp_cov)
        cov_lbcm_id = lbcm_bc(covs, np.eye(dim), lam_lbcm_id, C_refs=C_refs)
        nresults['lbcm_id'][t] = wass_dist(cov_lbcm_id, true_cov)

        # LBCM with barycenter base
        bc = true_bc(covs, np.ones(p) / p)
        lam_lbcm_bc, C_refs = opt_lam_lbcm(covs, bc, emp_cov)
        cov_lbcm_bc = lbcm_bc(covs, bc, lam_lbcm_bc, C_refs=C_refs)
        nresults['lbcm_bc'][t] = wass_dist(cov_lbcm_bc, true_cov)
            
    results[n] = nresults

print("Method   \t n = 64 \t\t n = 128 \t\t n = 256 \t\t n = 512")
methods = ['emp', 'grad_norm', 'lbcm_id', 'lbcm_bc']
method_name = {
    'emp'       : 'emp      ',
    'grad_norm' : 'grad_norm',
    'lbcm_id'   : 'lbcm_id  ',
    'lbcm_bc'   : 'lbcm_bc  '
}
for m in methods:
    print(method_name[m], end='\t')
    for n in nsamples:
        print(np.mean(results[n][m]), end='\t')
    print()
