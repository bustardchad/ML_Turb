#
# Converts large (~10 GB) HDF5 files to FITS files with only the relevant info

# Requires: HDF5 files named cr.out1.* (for gas and CR variables only, no need for user-created outputs)
# Returns: FITS file containing the 3D grid

import yt
from astropy.io import fits
import numpy as np
from tqdm import *
from skimage import exposure

# makeFits requires file directory and filename
# e.g. fileDir = "Files/MHD_beta1/"
#      fileName = "cr.out1.00040.athdf"
#
# Output: Writes a FITS file named turb.fits to file directory with density, B^2, and v_A fields


def makeFits(fileDir, fileName):
    # load in file with yt
    ds = yt.load(fileDir+fileName)

    grid = ds.r[::512j, ::512j, ::512j]

    fid_grid = grid.to_fits_data(
       # fields=[("gas", "density"), ("gas", "magnetic_energy_density"), ("gas", "alfven_speed")], length_unit=None)
        fields=[("athena_pp","Ec"), ("gas", "density"), ("gas", "magnetic_energy_density"), ("gas", "alfven_speed")], length_unit=None)

    fid_grid.writeto(fileDir+"turb_full.fits", overwrite=True)

    # test that the file wrote and has the correct stuff in it
    image_data = fits.open(fileDir+'turb_full.fits')
    print(image_data.info())

    print("Density header: ")
    print(image_data["density"].header)


# do this for all files I want

fileDir_arr = ['Files/CR_Diff_Fiducial_beta10/']
fileName_arr = ['cr.out1.00038.athdf']

i = 0
for fileDir in fileDir_arr:
    fileName = fileName_arr[i]
    print("Extracting: " + fileDir + fileName)
    makeFits(fileDir, fileName)
    i += 1
