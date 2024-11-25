import numpy as np

import matplotlib.pyplot as plt
import mnist

import ot

import images

#import tensorflow as tf
import time


#We include two methods to load and sort MNIST digits

# load and sort MNIST digits (no tensorflow)

mnist.temporary_dir = lambda: './mnist'

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()


#load and sort MNIST digits via tensorflow

#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


sorted_digits = {}
for i in range(10):
    sorted_digits[i] = []
    
for i in range(len(train_images)):
    label = train_labels[i]
    image = train_images[i]
    image = image / np.sum(image)
    
    sorted_digits[label] += [image]

# ====== picks out the images to use and applies corruption ======

ref_digit = 4
num_refs = 20

# NOTE to get the same result on every run, fix the permutation.
perm = np.random.permutation(len(sorted_digits[ref_digit]))
ref_inds = perm[:num_refs]

def corrupt(image):
    corrupted = image + 0
    corrupted[9:19,9:19] = 0.0
    corrupted = corrupted / corrupted.sum()
    return corrupted

ref_images = [sorted_digits[ref_digit][i] for i in ref_inds]
cref_images = [corrupt(ref) for ref in ref_images]


# ====== creates different base images ======

def create_base_image(ref_images, reg=0.005, cut_off=0.001):
    base_full = ot.bregman.convolutional_barycenter2d(ref_images,reg)
    base_trimmed = base_full * (base_full > cut_off)
    base_image = base_trimmed / base_trimmed.sum()
    return base_image

def create_checker_base():
    checkers = np.zeros((28,28))
    checkers[:14,:14] = 1
    checkers[14:,14:] = 1
    return checkers / checkers.sum()

def create_circle_base():
    circle = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            circle[i,j] = max(2 - np.abs(np.sqrt((i - 14) ** 2 + (j - 14) ** 2) - 10),0)
    return circle / circle.sum()

def create_double_checker_base():
    double_checker=np.zeros((28, 28))
    double_checker[0:4, 0:4] = 1
    double_checker[-4:, 0:4] = 1
    double_checker[0:4, -4:] = 1
    double_checker[-4:, -4:] = 1
    return double_checker/np.sum(double_checker)


base_1 = create_base_image(ref_images)
base_2 = np.ones((28,28)) / (28 * 28)
base_3 = create_checker_base()# chceckers
base_4 = create_circle_base()
base_5 = create_double_checker_base()


# ====== computes maps from bases to references ======

maps_1 = [images.compute_map(base_1, ref, reg=0.002) for ref in ref_images]
maps_2 = [images.compute_map(base_2, ref, reg=0.002) for ref in ref_images]
maps_3 = [images.compute_map(base_3, ref, reg=0.002) for ref in ref_images]
maps_4 = [images.compute_map(base_4, ref, reg=0.002) for ref in ref_images]
maps_5 = [images.compute_map(base_5, ref, reg=0.002) for ref in ref_images]


# ====== applies the corruption to the bases ======

cbase_1 = corrupt(base_1)
cbase_2 = corrupt(base_2)
cbase_3 = corrupt(base_3)
cbase_4 = corrupt(base_4)
cbase_5 = corrupt(base_5)

# ====== computes the maps between corruptoed images ======

cmaps_1 = [images.compute_map(cbase_1, cref, reg=0.002) for cref in cref_images]
cmaps_2 = [images.compute_map(cbase_2, cref, reg=0.002) for cref in cref_images]
cmaps_3 = [images.compute_map(cbase_3, cref, reg=0.002) for cref in cref_images]
cmaps_4 = [images.compute_map(cbase_4, cref, reg=0.002) for cref in cref_images]
cmaps_5 = [images.compute_map(cbase_5, cref, reg=0.002) for cref in cref_images]


# ====== make a figure ======

fig, axs = plt.subplots(6,7,figsize=(24,24))

for f in range(6):
    
    i = np.random.randint(500) +  num_refs 
    
    new_image = sorted_digits[ref_digit][i]
    cnew_image = corrupt(new_image)

    lbcm_lam_1, _, _, _ = images.find_coordinate(cbase_1, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_1)
    lbcm_lam_2, _, _, _ = images.find_coordinate(cbase_1, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_1)
    lbcm_lam_3, _, _, _ = images.find_coordinate(cbase_1, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_1)
    lbcm_lam_4, _, _, _ = images.find_coordinate(cbase_1, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_1)
    lbcm_lam_5, _, _, _ = images.find_coordinate(cbase_1, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_1)



    rec_1 = images.synthesize(base_1, ref_images, lbcm_lam_1, reg=0.001, ref_maps=maps_1)
    lbcm_image_1 = images.empirical_to_image(rec_1, lower_bound=0.0002)
    
    rec_2 = images.synthesize(base_2, ref_images, lbcm_lam_2, reg=0.001, ref_maps=maps_2)
    lbcm_image_2 = images.empirical_to_image(rec_2, lower_bound=0.0002)
    
    rec_3 = images.synthesize(base_3, ref_images, lbcm_lam_3, reg=0.001, ref_maps=maps_3)
    lbcm_image_3 = images.empirical_to_image(rec_3, lower_bound=0.0002)

    rec_4 = images.synthesize(base_4, ref_images, lbcm_lam_4, reg=0.001, ref_maps=maps_4)
    lbcm_image_4 = images.empirical_to_image(rec_4, lower_bound=0.0002)

    rec_5 = images.synthesize(base_5, ref_images, lbcm_lam_5, reg=0.001, ref_maps=maps_5)
    lbcm_image_5 = images.empirical_to_image(rec_5, lower_bound=0.0002)



    axs[f][0].imshow(new_image, cmap='binary')
    axs[f][1].imshow(cnew_image, cmap='binary')
    axs[f][2].imshow(lbcm_image_1, cmap='binary')
    axs[f][3].imshow(lbcm_image_2, cmap='binary')
    axs[f][4].imshow(lbcm_image_3, cmap='binary')
    axs[f][5].imshow(lbcm_image_4, cmap='binary')
    axs[f][6].imshow(lbcm_image_5, cmap='binary')


    axs[f][0].axis('off')
    axs[f][1].axis('off')
    axs[f][2].axis('off')
    axs[f][3].axis('off')
    axs[f][4].axis('off')
    axs[f][5].axis('off')
    axs[f][6].axis('off')

    
axs[0][0].set_title('Original', fontsize=20)
axs[0][1].set_title('Occluded', fontsize=20)
axs[0][2].set_title('LBCM (BC)', fontsize=20)
axs[0][3].set_title('LBCM (Uniform)', fontsize=20)
axs[0][4].set_title('LBCM (Checkers)', fontsize=20)
axs[0][5].set_title('LBCM (Circle)', fontsize=20)
axs[0][6].set_title('LBCM (Double Checker)', fontsize=20)

plt.tight_layout()
plt.show()
#plt.savefig('mnist_lbcm.pdf')