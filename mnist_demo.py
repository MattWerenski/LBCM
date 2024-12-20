# THIS SCRIPT MAY TAKE A FEW MINUTES TO CREATE THE IMAGE
# PLEASE GIVE IT SOME TIME, IT MAY LOOK LIKE ITS HANGING
# BUT THE PLOTTING IS A BIT SLOW

import matplotlib.pyplot as plt
import numpy as np



import images
import lbcm
import time
import ot

# THIS MAY NEED TO BE MODIFIED DEPENDING ON HOW YOU
# LOAD THE MNIST DATASET
#import tensorflow as tf
import mnist

# load and sort MNIST difits

mnist.temporary_dir = lambda: './mnist'

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Automatically downloads and loads the MNIST dataset
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# having a properly set train_images, train_labels, test_images and test_labels 
# is all that is required

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

base_1 = images.create_base_image(ref_images)

base_2=np.zeros((28, 28))
base_2[0:4, 0:4] = 1
base_2[-4:, 0:4] = 1
base_2[0:4, -4:] = 1
base_2[-4:, -4:] = 1
base_2=base_2/np.sum(base_2)

base_3 = np.ones((28,28)) / (28 * 28)


compute_maps_start_time_1=time.time()
maps_1 = [images.compute_map(base_1, ref, reg=0.00) for ref in ref_images]
cbase_1 = corrupt(base_1)
cmaps_1 = [images.compute_map(cbase_1, cref, reg=0.00) for cref in cref_images]
compute_maps_time_1=time.time()-compute_maps_start_time_1


compute_maps_start_time_2=time.time()
maps_2 = [images.compute_map(base_2, ref, reg=0.00) for ref in ref_images]
cbase_2 = corrupt(base_2)
cmaps_2 = [images.compute_map(cbase_2, cref, reg=0.00) for cref in cref_images]
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

    lbcm_lam_1, _, _, _ = images.find_coordinate(cbase_1, cref_images, cnew_image, reg=0.00, ref_maps=cmaps_1)
    lbcm_measure_1 = images.synthesize(base_1, ref_images, lbcm_lam_1, reg=0.00, ref_maps=maps_1)
    lbcm_image_1 = images.empirical_to_image(lbcm_measure_1)

    LBCM_1_total=time.time()-LBCM_1_start+compute_maps_time_1

    LBCM_2_start=time.time()

    lbcm_lam_2, _, _, _ = images.find_coordinate(cbase_2, cref_images, cnew_image, reg=0.00, ref_maps=cmaps_2)
    lbcm_measure_2 = images.synthesize(base_2, ref_images, lbcm_lam_2, reg=0.00, ref_maps=maps_2)
    lbcm_image_2 = images.empirical_to_image(lbcm_measure_2)

    LBCM_2_total=time.time()-LBCM_2_start+compute_maps_time_2
    
    LBCM_1_times.append(LBCM_1_total)
    LBCM_2_times.append(LBCM_2_total)

    #Linear reconstruction (no transport)

    linear_time_start=time.time()

    linear_lambda = images.linear_projection(cnew_image, np.array(ref_images))
    
    weighted_references=np.zeros(np.shape(ref_images[0]))
    for j in np.arange(len(linear_lambda)):
        weighted_references += linear_lambda[j] * ref_images[j]
    linear_time_total=time.time() - linear_time_start
    linear_times.append(linear_time_total)

    #BCM, initialized with linear reconstruction
    BCM_time_start = time.time()
    lam, _, _, _ = images.find_coordinate(cnew_image, cref_images, cnew_image, reg=0.00)
    bcm_measure = images.particle_synthesis(np.array(ref_images), lam,weighted_references, 200, 0.05, 0)
    bcm_image = images.empirical_to_image(bcm_measure)
    BCM_total_time = time.time() - BCM_time_start
    BCM_times.append(BCM_total_time)
    #Loss calculations 

    [original_mass, original_supp] = images.image_to_empirical(new_image)
    LBCM_distances_1 = ot.utils.dist(original_supp, lbcm_measure_1[1], metric='sqeuclidean') / 2
    lbcm_1_loss = ot.emd2(original_mass,lbcm_measure_1[0],LBCM_distances_1)
    LBCM_1_losses.append(lbcm_1_loss)
    LBCM_distances_2 = ot.utils.dist(original_supp, lbcm_measure_2[1], metric='sqeuclidean') / 2
    lbcm_2_loss = ot.emd2(original_mass,lbcm_measure_2[0],LBCM_distances_2)
    LBCM_2_losses.append(lbcm_2_loss)
    bcm_distances = ot.utils.dist(original_supp, bcm_measure[1], metric='sqeuclidean') / 2
    bcm_loss=ot.emd2(original_mass,bcm_measure[0],bcm_distances)
    BCM_losses.append(bcm_loss)

    [weighted_recon_mass, weighted_recon_supp] = images.image_to_empirical(weighted_references)
    linear_r_distances = ot.utils.dist(original_supp, weighted_recon_supp, metric='sqeuclidean') / 2
    linear_r_loss = ot.emd2(original_mass,weighted_recon_mass,linear_r_distances)
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



