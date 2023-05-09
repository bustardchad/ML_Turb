# Main file
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from astropy.io import fits
from skimage import exposure
import cmasher as cmr
import pickle
import pandas as pd
import glob
import random

# Function adapted from Peek and Burkhart 2019 (https://zenodo.org/record/2658177#.Y9fvEOzMJ24)
#
# Requires: fileDir -- file directory
#           fitsFileName -- file name
#           field -- e.g. 'density'
#           killPwr -- default is False
#
# Output: Creates and populates augmented data cube for a given FITS file
#         following the steps from Peek and Burkhart 2019:
#
# Steps: Create slices in the y-z plane (perpendicular to original B-field direction)
#        Extract nSlice=16 128 x 128 slices in each plane
#        Use periodic BCs to augment data -- shift 64 cells in y and z directions and redo slice extraction
#        Repeat for other snapshots, simulations, etc.



def pre_processing(fileDir, fitsFileName, field, killPwr=False, cuts = 6):
    # access data from FITS file
    fulldata = fits.open(fitsFileName)

    # data is logarithmic, so take the log
    data = np.log10(fulldata[field].data)

    # why multiply by 255 here??
    data = (data - np.nanmin(data)) / \
        (np.nanmax(data) - np.nanmin(data)) * 255.0

    # clean bad values (if any)
    data[~np.isfinite(data)] = 0.
    newdata = data

    # hardcoded image size, # of images in each direction, striding (for further augmentation)
    imsize, nx, nsh = 128, 4, 2
    dsh = data.shape


    # create data cube that will hold all slices
    # only training data has the augmentation (if nsh > 1)
    train_data = np.zeros([imsize, imsize, int((dsh[2]/cuts) * nsh * nsh * nx * nx)])
    val_data = np.zeros([imsize, imsize, int((dsh[2]/cuts) * nx * nx)])
    test_data = np.zeros([imsize, imsize, int((dsh[2]/cuts) * nx * nx)])

    iterct_train = 0
    iterct_val = 0
    iterct_test = 0
    if killPwr:
        for s in np.arange(dsh[0]):
            # slice in y-z plane
            img = data[s, :, :]

            # take 2D fft and normalize (kill the power spectrum)
            fftsl = np.fft.fft2(img)
            normalized = fftsl / np.abs(fftsl)
            newimg = np.real(np.fft.ifft2(normalized))
            newdata[s, :, :] = newimg

    # newdata holds the full set of slices, now we want to split into training and test sets
    for s in tqdm(np.arange(dsh[0])):
        if (s < (dsh[2]/cuts)-1): # only do augmentation on training data
            aug = nsh
        else:
            aug = 1

        sh = imsize / aug  # stride size depends on if we are augmenting
        for r in np.arange(aug):  # two loops that shift the image (augmentation)
            for q in np.arange(aug):
                fullimg = np.roll(np.roll(newdata[s, :, :], int(
                    r * sh + s), axis=0), int(q * sh + s), axis=1)
                for i in np.arange(nx):  # two loops that scan acros the 16 128 x 128 images
                    for j in np.arange(nx):
                        toimg = fullimg[int(
                            i * imsize):int((i + 1) * imsize), int(j * imsize):int((j + 1) * imsize)]

                        # equalize histogram
                        exeq = exposure.equalize_hist(
                            toimg)

                        # add slice to data cube depending on s value
                        if (s < (dsh[2]/cuts)-1):
                            train_data[:, :, iterct_train] = exeq
                            iterct_train += 1
                        elif ((s > (cuts-4.0)*dsh[2]/cuts) and (s < (cuts-3.0)*dsh[2]/cuts - 1)):
                            val_data[:, :, iterct_val] = exeq
                            iterct_val += 1
                        elif ((s > (cuts-2.0)*dsh[2]/cuts) and (s < (cuts-1.0)*dsh[2]/cuts - 1)):
                            test_data[:, :, iterct_test] = exeq
                            iterct_test += 1

    return train_data, val_data, test_data


# check a few random slices to make sure they look alright
def plot_random_slices(data):
    np.random.seed(0)
    random_indices = np.floor(np.random.random(3) * 2048)
    plt.figure(figsize=[16, 8])
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(data[:, :, int(random_indices[int(i)])], cmap='cmr.eclipse')
        plt.axis('off')
    plt.savefig('example_slices.pdf')


# Load in already created FITS files (using makeFits function in
# fromHDF5_to_cube.py)
#fileDirMHD_arr = ['CR_Advect_beta10', 'CR_Diff_Fiducial_beta10','CR_Diff100_beta10', 'CR_withStreaming_beta10']
#field_list = ['density','Ec']

fileDirMHD_arr = ['MHD_beta10']
field_list = ['density','magnetic_energy_density']

# also for sims with CRs, etc.
# fileDirC_arr = ['Files/MHD_beta1/']


# Start creating images!
killPwr = False


for fileDir in fileDirMHD_arr:

    sim_name = fileDir

    # get all .fits files
    files = glob.glob('../Files/'+fileDir+'/*.fits')

    # preprocess once per file directory to get length of datasets
    train_data, val_data, test_data = pre_processing(fileDir, files[0], field_list[0], killPwr, cuts=6)
    num_images_train = np.shape(train_data)[-1]
    num_images_val = np.shape(val_data)[-1]
    num_images_test = np.shape(test_data)[-1]

    # indices for random subsample
    ind_train = random.sample(range(0,num_images_train),int(num_images_train/4.))
    ind_val = random.sample(range(0,num_images_val),int(num_images_val/4.))
    ind_test = random.sample(range(0,num_images_test),int(num_images_test/4.))

    for field in field_list:

        # where to store full datasets, over multiple snapshots, for a given field
        ctr = 0
        for fitsFileName in files:


            # preprocess the whole set, dividing training and testing sets spatially by a buffer in the middle
            train_data, val_data, test_data = pre_processing(fileDir, fitsFileName, field, killPwr, cuts=6)

            # optionally plot random slices
            # plot_random_slices(train_data)


            # take a random subset of the array?
            train_data = train_data[:,:,ind_train]
            val_data = val_data[:,:,ind_val]
            test_data = test_data[:,:,ind_test]


            if (ctr==0):
                full_train = train_data
                full_val = val_data
                full_test = test_data
            else:
                # add to full datasets that include all snapshots
                full_train = np.concatenate((full_train,train_data),axis=2)
                full_val = np.concatenate((full_val,val_data),axis=2)
                full_test = np.concatenate((full_test,test_data),axis=2)

            ctr+=1

        print("Shape of full training set:" )
        print(full_train.shape)


        # Save files
        if killPwr:
            np.save('../Kill_Power/'+fileDir + f"/train_{sim_name}_{field}_small.npy", full_train)
            np.save('../Kill_Power/'+fileDir + f"/val_{sim_name}_{field}_small.npy", full_val)
            np.save('../Kill_Power/'+fileDir + f"/test_{sim_name}_{field}_small.npy", full_test)
        else:
            np.save('../Full_Power/'+fileDir + f"/train_{sim_name}_{field}_small.npy", full_train)
            np.save('../Full_Power/'+fileDir + f"/val_{sim_name}_{field}_small.npy", full_val)
            np.save('../Full_Power/'+fileDir + f"/test_{sim_name}_{field}_small.npy", full_test)

