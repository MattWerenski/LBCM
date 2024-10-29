import numpy as np
import scipy as sp
import scipy.stats

from gauss import *

dim = 10
p = 10
nsamples = [64, 128, 256, 512]
trials = 5 # plots were made with more replicates, this is to get results fast.
results = {}

for n in nsamples:
    print('nsamples = ',n)
    nresults = {
        'emp': np.zeros(trials),
        'bcm': np.zeros(trials),
        'lbcm_id': np.zeros(trials),
        'lbcm_bc': np.zeros(trials),
        'mle': np.zeros(trials),
        'bcm_lam': np.zeros(trials),
        'lbcm_id_lam': np.zeros(trials),
        'lbcm_bc_lam': np.zeros(trials),
        'mle_lam': np.zeros(trials)
    }
    
    t = 0
    while t < trials:
        print('trial ', t)
        try:
            
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
            

            # minimizes the norm of the gradient (which is done in the BCM)
            lam_bcm = np.array(opt_lam_grad_norm(covs, emp_cov)).squeeze()
            cov_bcm = true_bc(covs, lam_bcm)
            nresults['bcm'][t] = wass_dist(cov_bcm, true_cov)
            nresults['bcm_lam'][t] = np.sum(np.power(lam - lam_bcm,2))

            # LBCM with identity base
            lam_lbcm_id, C_refs = opt_lam_lbcm(covs, np.eye(dim), emp_cov)
            cov_lbcm_id = lbcm_bc(covs, np.eye(dim), lam_lbcm_id, C_refs=C_refs)
            nresults['lbcm_id'][t] = wass_dist(cov_lbcm_id, true_cov)
            nresults['lbcm_id_lam'][t] = np.sum(np.power(lam - lam_lbcm_id,2))

            # LBCM with barycenter base
            bc = true_bc(covs, np.ones(p) / p)
            lam_lbcm_bc, C_refs = opt_lam_lbcm(covs, bc, emp_cov)
            cov_lbcm_bc = lbcm_bc(covs, bc, lam_lbcm_bc, C_refs=C_refs)
            nresults['lbcm_bc'][t] = wass_dist(cov_lbcm_bc, true_cov)
            nresults['lbcm_bc_lam'][t] = np.sum(np.power(lam - lam_lbcm_bc,2))

            lam_mle = opt_lam_mle(covs, emp_cov)
            cov_mle = true_bc(covs, lam_mle)
            nresults['mle'][t] = wass_dist(cov_mle, true_cov)
            nresults['mle_lam'][t] = np.sum(np.power(lam - lam_mle,2))

            t += 1
        except e:
            # sometimes the MLE approach will fail to converge and raise an error.
            # replicate the trial if this occurs
            pass
            
    results[n] = nresults

print("W2 Error")
print("Method   \tn = 64 \t\tn = 128 \t\tn = 256 \t\tn = 512")
methods = ['emp', 'bcm', 'lbcm_id', 'lbcm_bc', 'mle']
method_name = {
    'emp'       : 'emp      ',
    'bcm'       : 'bcm      ',
    'lbcm_id'   : 'lbcm_id  ',
    'lbcm_bc'   : 'lbcm_bc  ',
    'mle'       : 'mle      ',
}
for m in methods:
    print(method_name[m], end='\t')
    for n in nsamples:
        print(np.mean(results[n][m]), end='\t')
    print()

print()
print("lambda Error")
print("Method   \tn = 64 \t\tn = 128 \t\tn = 256 \t\tn = 512")
methods = ['bcm', 'lbcm_id', 'lbcm_bc', 'mle']
method_name = {
    'bcm'       : 'bcm      ',
    'lbcm_id'   : 'lbcm_id  ',
    'lbcm_bc'   : 'lbcm_bc  ',
    'mle'       : 'mle      ',
}
for m in methods:
    print(method_name[m], end='\t')
    for n in nsamples:
        print(np.mean(results[n][m+'_lam']), end='\t')
    print()