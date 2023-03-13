# Main file
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from astropy.io import fits
from skimage import exposure
import cmasher as cmr
import pickle
import pandas as pd

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


def preProcessing(fileDir, field, killPwr=False):
    # access data from FITS file
    fulldata = fits.open('Files/'+fileDir + '/turb_full.fits')

    # data is logarithmic, so take the log
    data = np.log10(fulldata[field].data)

    # why multiply by 255 here??
    data = (data - np.nanmin(data)) / \
        (np.nanmax(data) - np.nanmin(data)) * 255.0

    # clean bad values (if any)
    data[~np.isfinite(data)] = 0.
    newdata = data

    # hardcoded image size, # of images in each direction, striding (for further augmentation)
    imsize, nx, nsh = 128, 4, 1
    sh = imsize / nsh  # stride size
    dsh = data.shape


    # create data cube that will hold all slices
    cube = np.zeros([imsize, imsize, int(dsh[2] * nsh * nsh * nx * nx)])
    iterct = 0
    if killPwr:
        for s in np.arange(dsh[0]):
            # slice in y-z plane
            img = data[s, :, :]

            # take 2D fft and normalize (kill the power spectrum)
            fftsl = np.fft.fft2(img)
            normalized = fftsl / np.abs(fftsl)
            newimg = np.real(np.fft.ifft2(normalized))
            newdata[s, :, :] = newimg

    for s in tqdm(np.arange(dsh[0])):
        for r in np.arange(nsh):  # two loops that shift the image (augmentation)
            for q in np.arange(nsh):
                fullimg = np.roll(np.roll(newdata[s, :, :], int(
                    r * sh + s), axis=0), int(q * sh + s), axis=1)
                for i in np.arange(nx):  # two loops that scan acros the 16 128 x 128 images
                    for j in np.arange(nx):
                        toimg = fullimg[int(
                            i * imsize):int((i + 1) * imsize), int(j * imsize):int((j + 1) * imsize)]

                        # equalize histogram
                        exeq = exposure.equalize_hist(
                            toimg)

                        # add slice to data cube
                        cube[:, :, iterct] = exeq
                        iterct += 1
    return cube


# check a few random slices to make sure they look alright
def plotRandomSlices(data):
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
#fileDirMHD_arr = ['MHD_beta1', 'MHD_beta10', 'MHD_beta100' ,'CR_Advect_beta10', 'CR_Diff_Fiducial_beta10','CR_Diff100_beta10']
fileDirMHD_arr = ['CR_Advect_beta10']
field_list = ['density']

# also for sims with CRs, etc.
# fileDirC_arr = ['Files/MHD_beta1/']


# Start creating images!
killPwr = True

for fileDir in fileDirMHD_arr:
    fulldata = fits.open('Files/' + fileDir + '/turb_full.fits')
    # preprocessing steps
    sim_name = fileDir
    for field in field_list:
        full_x = preProcessing(fileDir, field, killPwr)

        # plot random slices
        plotRandomSlices(full_x)


        # save full files (not broken into training, validation, and test sets yet)
        np.save('Files/'+fileDir + f"/data_{sim_name}_{field}_killPwr_noAugment.npy", full_x)

        #df = pd.DataFrame(full_x,columns=['index_i','index_j','image_id'])
        #df.to_pickle('Files/'+fileDir + f"/data_{sim_name}_{field}.pkl")
        # with open('Files/'+fileDir + f"/data_{sim_name}_{field}.pkl",'wb') as f1:
        #    pickle.dump(full_x,f1)
        # full_x.to_pickle('Files/'+fileDir + f"/data_{sim_name}_{field}.pkl")
