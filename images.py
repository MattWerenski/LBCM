# This file contains versions of the LBCM functions that are modified to work on images
# like those in the MNIST dataset. It also contains some basic methods for working
# with images.

import numpy as np
import ot
import scipy as sp
import scipy.stats
import cvxopt

import lbcm


# The parameters in this file have been tuned a bit to work well for MNIST images.
# It is mostly a wrapper for the lbcm file but tailored to handle images


def compute_map(source_image, target_image, reg=0.001):
    '''
    compute_map - Computes the entropic map from the base image to each reference image

    :param base_image: (m x n) np array corresponding to the image (not as an empirical measure)
    :param ref_image: list of np arrays corresponding to the reference images
    :param reg:        regularization parameter for the entropic map
    :return:           list of (m x 2) np arrays corresponding to the entropic maps from the base to the refs
    '''

    return lbcm.compute_map(
        image_to_empirical(source_image), 
        image_to_empirical(target_image), 
        reg=reg
    )
  
    
def compute_inner_products(base_image, ref_images, new_image, reg=0.001, ref_maps=[]):
    '''
    compute_inner_products - Computes the inner products used in the quadratic form

    :param base_image: base image in the LBCM
    :param ref_images: list of m reference images in the LBCM
    :param new_image:  new image to find the LBCM coordinates of
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] or leave empty to compute maps
    :return:           (m x m) array for the inner prodcuts, the entropic map for the new image,
                       and the maps from the base to the references.
    '''
    return lbcm.compute_inner_products(
        image_to_empirical(base_image),
        [image_to_empirical(ref) for ref in ref_images],
        image_to_empirical(new_image),
        reg=reg,
        ref_maps=ref_maps
    )


def find_coordinate(base_image, ref_images, new_image, reg=0.001, ref_maps=[]):
    '''
    find_coordinate - Finds the optimal coordinate in the LBCM

    :param base_image: base image in the LBCM
    :param ref_images: list of m reference images in the LBCM
    :param new_image:  new image to find the LBCM coordinates of
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] or leave empty to compute maps
    :return:           (m) np array corresponding to the estimated coordinate, (m x m) np array 
                       for the inner prodcuts, the entropic map for the new image, and the maps 
                       from the base to the references.
    '''
    return lbcm.find_coordinate(
        image_to_empirical(base_image),
        [image_to_empirical(ref) for ref in ref_images],
        image_to_empirical(new_image),
        reg=reg,
        ref_maps=ref_maps
    )

def synthesize(base_image, ref_images, lam, reg=0.001, ref_maps=[]):
    '''
    synthesize - Returns the image corresponding to the coordinate lam in the 
                 LBCM with given base and reference images
    :param base_image: base image in the LBCM
    :param ref_images: list of m reference images in the LBCM
    :param lam:        coordinate to estimate the image at
    :param reg:        regularization parameter in the entropic map
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] or leave empty to compute maps
    :return:           The [mass, support] pair for the synthesized measure
    '''
    
    return lbcm.synthesize(
        image_to_empirical(base_image),
        [image_to_empirical(ref) for ref in ref_images],
        lam,
        reg=reg,
        ref_maps=ref_maps
    )

def particle_synthesis(ref_images, lam, initial_image, iterations=200, step_size=0.05, reg=0.0):
    '''
    particle_synthesis - Particle method for estimating the barycenter of a set of measures
    
    :param ref_images:    list of images representing the references
    :param lam:           barycentric coordinate to estimate the barycenter of
    :param initial_image: starting measure for the fixed point process
    :param iterations:    number of fixed point iterations to run
    :param step_size:     how far to move at each iteration. Should be in (0,1)
    :param reg:           regularization parameter in the entropic map
    :return:              The [mass, support] pair for the synthesized measure
    '''
    return lbcm.particle_synthesis(
        [image_to_empirical(ref) for ref in ref_images],
        lam,
        image_to_empirical(initial_image),
        iterations=iterations, step_size=step_size, reg=reg
    )


def project(base_image, ref_images, new_image, reg=0.001, ref_maps=[]):
    '''
    synthesize_lbcm - Estimates the coordinate lambda and uses it to create the linear barycenter

    :param base_image: base image in the LBCM
    :param ref_images: list of m reference images in the LBCM
    :param new_image:  new image to find the LBCM coordinates of
    :param reg:        regularization parameter in the entropic map
    :param lam:        if provided, uses this coordinate instead of estimating it
    :param ref_maps:   if provided, uses these maps instead of re-computing them.
                       Pass [] to compute maps
    :return:           empirical linear barycenter as a [mass, support] pair,
                       (m x m) np array for the inner prodcuts, the entropic map for the new image, 
                       and the maps from the base to the references.
    '''

    return lbcm.project(
        image_to_empirical(base_image),
        [image_to_empirical(ref) for ref in ref_images],
        image_to_empirical(new_image),
        reg=reg,
        ref_maps=ref_maps
    )


def linear_projection(to_project, refs):
    '''
    linear_projection - Estimates the coordinate using a linear projection onto the references
        which essentially treats all the images as vectors in a R^764 (28 * 28 = 764)
        
    :param to_project: image to be linearly projected onto the convex hull of the references
    :param refs:       list of images to linearly project onto
    :return:           np.array of the same length as refs which gives the convex combination
                       of the refs which is closest to to_project. The projection can be recovered
                       by taking a weighted some of the refs with the returned vector as weights.
    '''
    
    # flattens the images into vectors (if necessary)
    to_project = np.array(to_project)
    if len(to_project.shape) > 1:
        to_project = to_project.reshape(-1)
        
    refs = np.array(refs)
    if len(refs.shape) > 2:
        refs = refs.reshape((refs.shape[0],-1))
    
    # sets up a quadratic program for projection onto the convex hull
    m = refs.shape[0]
    P = cvxopt.matrix(refs @ refs.T)
    G = cvxopt.matrix(-np.eye(m))
    h = cvxopt.matrix(np.zeros(m))
    q = cvxopt.matrix(-to_project @ refs.T)
    A = cvxopt.matrix(np.ones((1,m)))
    b = cvxopt.matrix(np.ones((1,1)))
    soln = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    lam = np.squeeze(np.array(soln['x']))
    return lam


def create_base_image(ref_images, reg=0.005, cut_off=0.001):
    '''
    create_base_image - Generates an average using a convolution barycenter of the references

    :param ref_images: list of (n x m) np arrays which should sum to 1 and be non-negative
    :param reg:        regularization paramter for the convolution
    :param cut_off:    float, pixels below this intensity are set to 0
    :return:           (n x m) np array containing the convolution
    '''
    base_full = ot.bregman.convolutional_barycenter2d(ref_images, reg)
    base_trimmed = base_full * (base_full > cut_off)
    base_image = base_trimmed / base_trimmed.sum()
    return base_image

def image_to_empirical(image:np.array):
    '''
    image_to_empirical - Converts an image into an empirical measure which tracks support and mass
                         (the mass is not normalized to a total of 1, but the coordinates are in [0,1]^2)
    
    :param image: (n x m) np array representing an image
    :return:      [(l) np array of mass, (l x 2) np array of the support location]
    '''

    [height, width] = image.shape

    # for normalizing the height to be between 0 and 1
    # handles the edge case of height or width being 1 pixel
    nheight = max(height - 1, 1) 
    nwidth = max(width - 1, 1)

    support = []
    mass = []
    for i in range(height):
        for j in range(width):
            if image[i,j] == 0:
                continue
                
            support += [[i /  nheight, j / nwidth]]
            mass += [image[i,j]]
            
    return [np.array(mass) , np.array(support)]

def empirical_to_image(measure, height=28, width=28, 
                       resolution=5, lower_bound=0.0002, bw_method=0.15):
    '''
    empirical_to_image - Converts an empirical measure into an image

    :param measure:     [mass, support] pair representing the measure    
    :param height:      positive integer, desired output height
    :param width:       positive integer, desired output width
    :param resolution:  positive integer, upscaling to use in the KDE
    :param lower_bound: float, pixels below this intensity are set to 0
    :bw_method:         bw_method parameter used in scipy.stats.gaussian_kde
    :return:            (height x width) np array containing the image
    '''

    [mass, support] = measure
    
    kde = sp.stats.gaussian_kde(support.T, bw_method=bw_method, weights=mass)

    # creates a 2D grid of locations to evaluate the KDE at
    grid = np.array(
        np.meshgrid(
            np.linspace(0, 1, height * resolution),
            np.linspace(0, 1, width * resolution)
        )
    )
    mesh = grid.reshape(2, height * resolution * width * resolution)

    density = kde(mesh).reshape(height * resolution, width *resolution).T
    density = density / density.sum()
    density = density > lower_bound

    # reduces the resolution to the target amount
    blur = np.zeros((height,width))
    for i in range(resolution):
        for j in range(resolution):
            blur += density[i::resolution, j::resolution]
    
    return blur / blur.sum()





