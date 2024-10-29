import numpy as np
import ot
from lbcm import *

def compute_maps(base_image:np.array, ref_images, reg):

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

def particle_synthesis(ref_images,weights,initial_image,iterations,stepsize,reg):
    m=len(ref_images)
    ref_supports=[]
    ref_masses=[]
    for ref_image in ref_images:
        ref_support, ref_mass = image_to_empirical(ref_image)
        ref_supports.append(ref_support)
        ref_masses.append(ref_mass)
    initial_support,initial_mass=image_to_empirical(initial_image)
    for i in np.arange(iterations):
        weighted_maps=[]
        for j in np.arange(m):
            dist = ot.utils.dist(initial_support, ref_supports[j], metric='sqeuclidean') / 2
            if reg == 0:
                coupling = ot.lp.emd(initial_mass, ref_masses[j], dist)
            else:
                coupling = ot.sinkhorn(initial_mass, ref_masses[j], dist, reg)
            weighted_maps += [np.dot(weights[j],coupling_to_map(coupling, ref_supports[j]))]
        map=np.sum(weighted_maps,axis=0)
        initial_support=np.dot((1-stepsize),initial_support)+np.dot(stepsize,map)
    return initial_support,initial_mass
