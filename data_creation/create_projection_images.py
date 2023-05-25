# Main file
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from astropy.io import fits
from skimage import exposure
import cmasher as cmr
import pandas as pd
import glob
import random

# Adapts and builds upon the data creation pipeline from Peek and Burkhart 2019
#   See: (https://zenodo.org/record/2658177#.Y9fvEOzMJ24)
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


######################################################
# Helper functions
def project_image(img,s,depth):
    # Returns a projection (average) along axis 0 of input image (img)
    # between index s and s+depth
    return np.average(img[s:s+depth,:,:],axis=0)


def flat(img):
    # Returns a new img with power spectra flattened
    fftsl = np.fft.fft2(img)
    normalized = fftsl / np.abs(fftsl)
    newimg = np.real(np.fft.ifft2(normalized))
    return newimg


def find_zero_images(data):
    # Finds and returns all indices of 3D image array where images are all zeros
    zero_indices = []
    for i in range(data.shape[-1]):
        if np.any(np.all((data[:,:,i] <= 0))):
            zero_indices.append(i)
    return zero_indices


def plot_random_slices(data):
# check a few random slices to make sure they look alright
    np.random.seed(0)
    random_indices = np.floor(np.random.random(3) * 2048)
    plt.figure(figsize=[16, 8])
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(data[:, :, int(random_indices[int(i)])], cmap='cmr.eclipse')
        plt.axis('off')
    plt.savefig('example_slices.pdf')

#########################################################

def pre_processing(fileDir, fitsFileName, field, killPwr=False, cuts = 6, depth=8):
    # Main preprocessing function
    #
    # Inputs:
    #   fileDir = directory of FITS files we want to process
    #
    #   fitsFileName = Name of FITS files -- can be specific names, or use *.fits to process all of them
    #
    #   field = variable to create images of. Can be any of "density", "magnetic_energy_density", "Ecr"
    #
    #   killPwr (default = False) = flag for whether or not to flatten the power spectrum of the image
    #
    #   cuts (default = 6) = how many separate regions to create in each data cube.
    #
    #       Notes: Spatially correlated turbulence structures present a multitute of problems.
    #              -- Structures present in the training set can bleed into the validation set
    #              -- Can make training images correlated to each other, therefore easy for the CNN to "cheat"
    #
    #       Solution: The training data is the first 1/6th of a data cube, followed by a buffer, then the validation
    #                   set is the next 1/6th of the cube, and so on
    #
    #   depth (default = 8) = how many cells to average over in the x-direction. depth = 1 corresponds to a simple
    #                       slice plot
    #
    # Tasks:
    #   1. Load data cubes stored in FITS files and clean bad values, if any
    #   2.
    #

    # access data from FITS file
    fulldata = fits.open(fitsFileName)

    # data is logarithmic, so take the log
    data = np.log10(fulldata[field].data)

    data = (data - np.nanmin(data)) / \
        (np.nanmax(data) - np.nanmin(data)) * 255.0

    # clean bad values (if any)
    data[~np.isfinite(data)] = 0.
    #newdata = np.zeros(int(data.shape[0]/depth),data.shape[1],data.shape[2])

    # hardcoded image size, # of images in each direction, striding (for further augmentation)
    imsize, nx, nsh = 128, 4, 2
    dsh = data.shape


    # Calculate number of training, validation, and test images to hold in each array
    num_train_samples = int((dsh[2]/cuts) * nsh * nsh * nx * nx/depth) + 1
    num_val_samples = int((dsh[2]/cuts) * nsh * nsh * nx * nx/depth) + 1
    num_test_samples = int((dsh[2]/cuts) * nsh * nsh * nx * nx/depth) + 1

    """
    print("Number of training images: " + str(num_train_samples))
    print("Number of val images: " + str(num_val_samples))
    print("Number of test images: " + str(num_test_samples))
    """

    # create 3D arrays that will hold all 2D images -- one cube each for training, val, and test sets
    train_data = np.zeros([imsize, imsize, num_train_samples])
    val_data = np.zeros([imsize, imsize, num_val_samples])
    test_data = np.zeros([imsize, imsize, num_test_samples])

    """
    train_reg_max = dsh[2]/cuts - 1
    val_reg_min = (cuts-4.0)*dsh[2]/cuts
    val_reg_max = (cuts-3.0)*dsh[2]/cuts - 1
    test_reg_min = (cuts-2.0)*dsh[2]/cuts
    test_reg_max = (cuts-1.0)*dsh[2]/cuts - 1


    print("Size of training data region: " + str(train_reg_max))
    print("Size of validation data region: " + str(val_reg_max - val_reg_min))
    print("Size of test data region: " + str(test_reg_max - test_reg_min))
    """

    iterct_train = 0
    iterct_val = 0
    iterct_test = 0



    # Begin scrolling through each data cube
    for s in tqdm(np.arange(0,dsh[0],depth)): # s is x-index for original data cube

        # For projection plots, averaged over depth > 1 # of cells, we are decreasing the amount of
        # training, validation, and test data considerably compared to the depth = 1 slices, so I've
        # commented out the following lines s.t. each dataset is augmented instead of just the train set
        """
        if (s < (dsh[2]/cuts)-depth): # only do augmentation on training data
            aug = nsh
        else:
            aug = 1
        """
        aug = nsh
        # Return a (possibly flattened) projection of size 512 x 512
        if killPwr:
            # take projection AND flatten power spectrum
            input_projection = flat(project_image(data,s,depth))
        else:
            # take projection
            input_projection = project_image(data,s,depth)

        sh = imsize / aug  # stride size depends on if we are augmenting

        # Data augmentation taking advantage of periodic boundary conditions in simulation box
        for r in np.arange(aug):  # two loops that shift the 512x512 image
            for q in np.arange(aug):
                # roll along axes of 512x512 image
                fullimg = np.roll(np.roll(input_projection, int(
                    r * sh + s), axis=0), int(q * sh + s), axis=1)

                for i in np.arange(nx):  # two loops that scan acros the 16 128 x 128 images
                    for j in np.arange(nx):
                        toimg = fullimg[int(
                            i * imsize):int((i + 1) * imsize), int(j * imsize):int((j + 1) * imsize)]

                        # equalize histogram using astropy functionality
                        exeq = exposure.equalize_hist(
                            toimg)


                        # add slice to data cube depending on s value
                        if (s < (dsh[2]/cuts) - depth-1):
                            train_data[:, :, iterct_train] = exeq
                            iterct_train += 1
                        elif ((s > (cuts-4.0)*dsh[2]/cuts) and (s < (cuts-3.0)*dsh[2]/cuts - depth-1)):
                            val_data[:, :, iterct_val] = exeq
                            iterct_val += 1
                        elif ((s > (cuts-2.0)*dsh[2]/cuts) and (s < (cuts-1.0)*dsh[2]/cuts - depth-1)):
                            test_data[:, :, iterct_test] = exeq
                            iterct_test += 1


    return train_data, val_data, test_data

###################################################


# Load in already created FITS files (using makeFits function in
# fromHDF5_to_cube.py)

#fileDirMHD_arr = ['CR_Advect_beta10', 'CR_Diff_Fiducial_beta10','CR_Diff100_beta10', 'CR_withStreaming_beta10']
#field_list = ['density','Ec']

fileDirMHD_arr = ['MHD_beta10']
field_list = ['density','magnetic_energy_density']



# Start creating images!
killPwr = False
depth = 8 # how many images to average over in the x-direction

for fileDir in fileDirMHD_arr:

    sim_name = fileDir

    # get all .fits files
    files = glob.glob('../Files/'+fileDir+'/*.fits')

    # preprocess once per file directory to get length of datasets
    train_data, val_data, test_data = pre_processing(fileDir, files[0], field_list[0], killPwr, cuts=6, depth=depth)

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
            train_data, val_data, test_data = pre_processing(fileDir, fitsFileName, field, killPwr, cuts=6, depth=depth)

            # Find indices of images that remain totally zero-valued (didn't fully fill in train_data, val_data, and
            # test_data arrays)
            zero_indices_train = find_zero_images(train_data)
            zero_indices_val = find_zero_images(val_data)
            zero_indices_test = find_zero_images(test_data)

            # Limit train, val, and test data to only images that are non-zero

            #print("Training images with all zeros: " + str(zero_indices_train))
            train_data = train_data[:,:,:min(zero_indices_train)-1]
            #print("Length of training data cube with all zero images cut out: " + str(train_data.shape[-1]))


            #print("Validation images with all zeros: " + str(zero_indices_val))
            val_data = val_data[:,:,:min(zero_indices_val)-1]
            #print("Length of validation data cube with all zero images cut out: " + str(val_data.shape[-1]))

            #print("Test images with all zeros: " + str(zero_indices_test))
            test_data = test_data[:,:,:min(zero_indices_test)-1]
            #print("Length of test data cube with all zero images cut out: " + str(test_data.shape[-1]))

            # optionally plot random slices
            #plot_random_slices(train_data)


            # TODO: Put in a flag for this

            # If uncommented, takes a random subsample, without replacement, to give 1/4 of the images
            # These smaller datasets are referenced with the word "small" in their names
            """
            # take a random subset of the array?
            train_data = train_data[:,:,ind_train]
            val_data = val_data[:,:,ind_val]
            test_data = test_data[:,:,ind_test]
            """


            # Add images for each file to a larger 3D array
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


        # Save full 3D arrays to .npy files
        if killPwr:
            np.save('../Kill_Power/'+fileDir + f"/train_{sim_name}_{field}_depth_{depth}_small.npy", full_train)
            np.save('../Kill_Power/'+fileDir + f"/val_{sim_name}_{field}_depth_{depth}_small.npy", full_val)
            np.save('../Kill_Power/'+fileDir + f"/test_{sim_name}_{field}_depth_{depth}_small.npy", full_test)
        else:
            np.save('../Full_Power/'+fileDir + f"/train_{sim_name}_{field}_depth_{depth}_small.npy", full_train)
            np.save('../Full_Power/'+fileDir + f"/val_{sim_name}_{field}_depth_{depth}_small.npy", full_val)
            np.save('../Full_Power/'+fileDir + f"/test_{sim_name}_{field}_depth_{depth}_small.npy", full_test)
