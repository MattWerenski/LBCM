import matplotlib.pyplot as plt
import mnist
import numpy as np

from images import create_base_image, empirical_to_image
from lbcm import *
import mnist_utilities


mnist.temporary_dir = lambda: './mnist'

# load and sort MNIST difits

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

sorted_digits = {}
for i in range(10):
    sorted_digits[i] = []
    
for i in range(len(train_images)):
    label = train_labels[i]
    image = train_images[i]
    image = image / np.sum(image)
    
    sorted_digits[label] += [image]

# creates reference images and the corrupted versions

ref_digit = 4
num_refs = 20
perm = np.random.permutation(len(sorted_digits[ref_digit]))
ref_inds = perm[:num_refs]

def corrupt(image):
    corrupted = image + 0
    corrupted[9:19,9:19] = 0.0
    corrupted = corrupted / corrupted.sum()
    return corrupted

ref_images = [sorted_digits[ref_digit][i] for i in ref_inds]
cref_images = [corrupt(ref) for ref in ref_images]

# set up the base measures for the LBCM's

base_1 = create_base_image(ref_images)
base_2 = np.ones((28,28)) / (28 * 28)

maps_1 = compute_maps(base_1, ref_images, reg=0.002)
maps_2 = compute_maps(base_2, ref_images, reg=0.002)

cbase_1 = corrupt(base_1)
cbase_2 = corrupt(base_2)

cmaps_1 = compute_maps(cbase_1, cref_images, reg=0.002)
cmaps_2 = compute_maps(cbase_2, cref_images, reg=0.002)

# creates a plot

fig, axs = plt.subplots(6,5,figsize=(10,12))

for f in range(6):
    
    i = np.random.randint(500) +  num_refs 
    
    new_image = sorted_digits[ref_digit][i]
    cnew_image = corrupt(new_image)

    lbcm_lam_1, _, _, _ = find_coordinate_lbcm(cbase_1, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_1)
    lbcm_lam_2, _, _, _ = find_coordinate_lbcm(cbase_2, cref_images, cnew_image, reg=0.002, ref_maps=cmaps_2)

    rec_support_1, rec_mass_1 = synthesize_lbcm(base_1, None, None, reg=0.002, lam=lbcm_lam_1, ref_maps=maps_1)
    rec_support_2, rec_mass_2 = synthesize_lbcm(base_2, None, None, reg=0.002, lam=lbcm_lam_2, ref_maps=maps_2)

    lbcm_image_1 = empirical_to_image(rec_support_1, rec_mass_1)
    lbcm_image_2 = empirical_to_image(rec_support_2, rec_mass_2)

    ip = mnist_utilities.inner_products(cnew_image, np.array(cref_images))
    lam = mnist_utilities.solve(ip)
    bcm_image = mnist_utilities.convolutional_barycenter(np.array(ref_images), lam)

    axs[f][0].imshow(new_image, cmap='binary')
    axs[f][1].imshow(cnew_image, cmap='binary')
    axs[f][2].imshow(lbcm_image_1, cmap='binary')
    axs[f][3].imshow(lbcm_image_2, cmap='binary')
    axs[f][4].imshow(bcm_image, cmap='binary')

    axs[f][0].axis('off')
    axs[f][1].axis('off')
    axs[f][2].axis('off')
    axs[f][3].axis('off')
    axs[f][4].axis('off')
    
axs[0][0].set_title('Original', fontsize=20)
axs[0][1].set_title('Occluded', fontsize=20)
axs[0][2].set_title('LBCM (BC)', fontsize=20)
axs[0][3].set_title('LBCM (Uniform)', fontsize=20)
axs[0][4].set_title('BCM', fontsize=20)

plt.tight_layout()
plt.show()



