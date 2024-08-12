import numpy as np
import ot
import scipy as sp
import scipy.stats

# the parameters in this file have been tuned a bit to work well for MNIST images

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
    :return:      (l x 2) np array of the support location and (l) np array of mass
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
            
    return np.array(support), np.array(mass)

def empirical_to_image(support:np.array, mass:np.array, height=28, width=28, 
                       resolution=5, lower_bound=0.0002, bw_method=0.15):
    '''
    empirical_to_image - Converts an empirical measure into an image

    :param support:     (l x 2) np arrayo fo the support loacations
    :param mass:        (l) np array of the mass at each location
    :param height:      positive integer, desired output height
    :param width:       positive integer, desired output width
    :param resolution:  positive integer, upscaling to use in the KDE
    :param lower_bound: float, pixels below this intensity are set to 0
    :bw_method:         bw_method parameter used in scipy.stats.gaussian_kde
    :return:            (height x width) np array containing the image
    '''

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





