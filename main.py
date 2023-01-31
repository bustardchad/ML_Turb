# Main file
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
from astropy.io import fits
from skimage import exposure
import cmasher as cmr

# Function adapted from Peek and Burkhart 2019 (https://zenodo.org/record/2658177#.Y9fvEOzMJ24)
#
# Requires: fileDir, fitsFileName (turb_full.fits, unless something gets changed), killPwr flag
#
# Output: Creates and populates augmented data cube for a given FITS file following the steps from Peek     and Burkhart 2019:
#   Create slices in the y-z plane (perpendicular to original B-field direction)
#   Extract nSlice=16 128 x 128 slices in each plane
#   Use periodic BCs to augment data -- shift 64 cells in y and z directions and redo slice extraction,     add to dataset
#
#   Repeat for other snapshots, simulations, etc.
def preProcessing(fileDir,fileName,field,nx,killPwr=False):
    fulldata = fits.open(fileDir+'turb_full.fits')
    data = fulldata[field].data
    # why multiply by 255 here??
    data = (data- np.nanmin(data))/(np.nanmax(data)- np.nanmin(data))*255.0
    data[~np.isfinite(data)] = 0.

    imsize,nsh = 128, 2  # hardcoded imagize size, striding, and number of images across a cube
    sh = imsize/nsh # stride size
    dsh = data.shape
    print(dsh)
    newdata = np.log10(data)
    cube = np.zeros([imsize, imsize, int(dsh[2]*nsh*nsh*nx*nx)])
    iterct = 0
    for s in np.arange(dsh[0]):
        if killPwr:
            img = data[s, :, :] # slice in y-z plane
            fftsl= np.fft.fft2(img)
            normalized = fftsl/np.abs(fftsl)
            newimg = np.real(np.fft.ifft2(normalized))
            newdata[s, :, :] = newimg
    for s in tqdm(np.arange(dsh[2])):
        for r in np.arange(nsh):
            for q in np.arange(nsh):
                fullimg = np.roll(np.roll(newdata[s, :, :], int(r*sh+s), axis=0), int(q*sh+s), axis=1)
                for i in np.arange(nx):
                     for j in np.arange(nx):
                        toimg = fullimg[int(i*imsize):int((i+1)*imsize), int(j*imsize):int((j+1)*imsize)]
                        exeq = exposure.equalize_hist(toimg) # equalize histogram
                        cube[:, :, iterct] = exeq
                        #plt.imshow(toimg)
                        #print(np.max(toimg))
                       # exeq = exeq.reshape([imsize, imsize, 1])*np.ones([imsize, imsize, 3])*255
                       # img = Image.fromarray(exeq.astype('int8'), mode='RGB')
                        iterct+=1
    return cube # note: contains all fields, i.e. density, B^2, v_A, etc.

def plotRandomSlices(data):
    # check a few random slices to make sure they look alright
    np.random.seed(0)
    random_indices = np.floor(np.random.random(3)*2048)
    plt.figure(figsize=[16, 8])
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(data[:, :, int(random_indices[int(i)])], cmap='cmr.eclipse')
        plt.axis('off')
    plt.savefig('example_slices.pdf')



# Load in already created FITS files (using makeFits function in fromHDF5_to_cube.py)
fileDirMHD_arr = ['Files/MHD_beta1/']
fileNameMHD_arr = ['cr.out1.00040.athdf']

# also for sims with CRs, etc.
#fileDirC_arr = ['Files/MHD_beta1/']
#fileNameMHD_arr = ['cr.out1.00040.athdf']


# Start with MHD simulations
nx = 1

i = 0
for fileDir in fileDirMHD_arr:
    fileName = fileNameMHD_arr[i]
    killPwr = False

    # preprocessing steps
    field = 'density'
    full_x = preProcessing(fileDir,fileName,field,nx,killPwr)

    # plot random slices
    plotRandomSlices(full_x)


    # split into training and test sets, save as .npy files
    train_x = full_x[0:96, :, :]
    np.save('train_MHD_00040.npy', train_x)  # npy format is faster to read than csv, txt, etc.

    test_x = full_x[(128-16-8):-8, :, :]
    np.save('test_MHD_00040.npy', test_x)  # npy format is faster to read than csv, txt, etc.

    i+=1


