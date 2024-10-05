import matplotlib.pyplot as plt
import mnist
import numpy as np

from images import create_base_image, empirical_to_image
from lbcm import *
import mnist_utilities
import synthesis
import tensorflow as tf
import time

#mnist.temporary_dir = lambda: './mnist'

# load and sort MNIST difits

#train_images = mnist.train_images()
#train_labels = mnist.train_labels()

#test_images = mnist.test_images()
#test_labels = mnist.test_labels()

# Automatically downloads and loads the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

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

base_2=np.zeros((28, 28))
base_2[0:4, 0:4] = 1
base_2[-4:, 0:4] = 1
base_2[0:4, -4:] = 1
base_2[-4:, -4:] = 1
base_2=base_2/np.sum(base_2)

base_3 = np.ones((28,28)) / (28 * 28)


compute_maps_start_time_1=time.time()
maps_1 = compute_maps(base_1, ref_images, reg=0.00)
cbase_1 = corrupt(base_1)
cmaps_1 = compute_maps(cbase_1, cref_images, reg=0.00)
compute_maps_time_1=time.time()-compute_maps_start_time_1


compute_maps_start_time_2=time.time()
maps_2 = compute_maps(base_2, ref_images, reg=0.00)
cbase_2 = corrupt(base_2)
cmaps_2 = compute_maps(cbase_2, cref_images, reg=0.00)
compute_maps_time_2=time.time()-compute_maps_start_time_2



# creates a plot

fig, axs = plt.subplots(6,6,figsize=(10,12))
LBCM_1_losses=[]
LBCM_2_losses=[]
BCM_losses=[]
linear_losses=[]

LBCM_1_times=[]
LBCM_2_times=[]
linear_times=[]
BCM_times=[]

for f in range(6):
    
    i = np.random.randint(500) +  num_refs 

    #LBCM
    new_image = sorted_digits[ref_digit][i]
    cnew_image = corrupt(new_image)
    LBCM_1_start=time.time()

    lbcm_lam_1, _, _, _ = find_coordinate_lbcm(cbase_1, cref_images, cnew_image, reg=0.00, ref_maps=cmaps_1)
    rec_support_1, rec_mass_1 = synthesize_lbcm(base_1, None, None, reg=0.00, lam=lbcm_lam_1, ref_maps=maps_1)
    lbcm_image_1 = empirical_to_image(rec_support_1, rec_mass_1)

    LBCM_1_total=time.time()-LBCM_1_start+compute_maps_time_1

    LBCM_2_start=time.time()

    lbcm_lam_2, _, _, _ = find_coordinate_lbcm(cbase_2, cref_images, cnew_image, reg=0.00, ref_maps=cmaps_2)
    rec_support_2, rec_mass_2 = synthesize_lbcm(base_2, None, None, reg=0.00, lam=lbcm_lam_2, ref_maps=maps_2)
    lbcm_image_2 = empirical_to_image(rec_support_2, rec_mass_2)

    LBCM_2_total=time.time()-LBCM_2_start+compute_maps_time_2
    
    LBCM_1_times.append(LBCM_1_total)
    LBCM_2_times.append(LBCM_2_total)

    #Linear reconstruction (no transport)

    linear_time_start=time.time()

    linear_lambda=mnist_utilities.linear_projection(cnew_image, np.array(ref_images))
    weighted_references=np.zeros(np.shape(ref_images[0]))
    for j in np.arange(len(linear_lambda)):
        weighted_references=weighted_references+np.dot(linear_lambda[j],ref_images[j])
    linear_time_total=time.time()-linear_time_start
    linear_times.append(linear_time_total)

    #BCM, initialized with linear reconstruction
    BCM_time_start=time.time()
    ip = mnist_utilities.inner_products(cnew_image, np.array(cref_images))
    lam = mnist_utilities.solve(ip)
    bcm_support,bcm_mass=synthesis.particle_synthesis(np.array(ref_images),lam,weighted_references,200,0.05,0)
    bcm_image=empirical_to_image(bcm_support,bcm_mass)
    BCM_total_time=time.time()-BCM_time_start
    BCM_times.append(BCM_total_time)
    #Loss calculations 

    original_supp,original_mass=image_to_empirical(new_image)
    LBCM_distances_1= ot.utils.dist(original_supp, rec_support_1, metric='sqeuclidean') / 2
    lbcm_1_loss=ot.emd2(original_mass,rec_mass_1,LBCM_distances_1)
    LBCM_1_losses.append(lbcm_1_loss)
    LBCM_distances_2=ot.utils.dist(original_supp, rec_support_2, metric='sqeuclidean') / 2
    lbcm_2_loss=ot.emd2(original_mass,rec_mass_2,LBCM_distances_2)
    LBCM_2_losses.append(lbcm_2_loss)
    bcm_distances=ot.utils.dist(original_supp, bcm_support, metric='sqeuclidean') / 2
    bcm_loss=ot.emd2(original_mass,bcm_mass,bcm_distances)
    BCM_losses.append(bcm_loss)

    weighted_recon_supp, weighted_recon_mass=image_to_empirical(weighted_references)
    linear_r_distances=ot.utils.dist(original_supp, weighted_recon_supp, metric='sqeuclidean') / 2
    linear_r_loss=ot.emd2(original_mass,weighted_recon_mass,linear_r_distances)
    linear_losses.append(linear_r_loss)

    axs[f][0].imshow(new_image, cmap='binary')
    axs[f][1].imshow(cnew_image, cmap='binary')
    axs[f][2].imshow(lbcm_image_1, cmap='binary')
    axs[f][3].imshow(lbcm_image_2, cmap='binary')
    axs[f][4].imshow(bcm_image, cmap='binary')
    axs[f][5].imshow(weighted_references,cmap='binary')

    axs[f][0].axis('off')
    axs[f][1].axis('off')
    axs[f][2].axis('off')
    axs[f][3].axis('off')
    axs[f][4].axis('off')
    axs[f][5].axis('off')
    #axs[f][5].imshow
mean_LBCM_1_time=np.mean(LBCM_1_times)
mean_LBCM_2_time=np.mean(LBCM_2_times)
mean_linear_time=np.mean(linear_times)
mean_BCM_time=np.mean(BCM_times)
axs[0][0].set_title('Original', fontsize=20)
axs[0][1].set_title('Occluded', fontsize=20)
axs[0][2].set_title('LBCM (BC)\n Avg loss:{:.5f}\n Avg time:{:.4f}'.format(np.mean(LBCM_1_losses),mean_LBCM_1_time), fontsize=20)
axs[0][3].set_title('LBCM (Exotic)\n Avg loss:{:.5f}\n Avg time:{:.4f}'.format(np.mean(LBCM_2_losses),mean_LBCM_2_time), fontsize=20)
axs[0][4].set_title('BCM\n Avg loss:{:.5f}\n Avg time:{:.4f}'.format(np.mean(BCM_losses),mean_BCM_time), fontsize=20)
axs[0][5].set_title('Linear Reconstruction\n Avg loss:{:.5f}\n Avg time{:.4f}'.format(np.mean(linear_losses),mean_linear_time), fontsize=20)



plt.tight_layout()
plt.show()



