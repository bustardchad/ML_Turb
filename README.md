# Deep Learning the Effects of Cosmic Rays on Astrophysical Turbulence
## (Chad Bustard and John Wu, in prep)


### These files extract and augment image data from MHD turbulent box simulations (from Bustard and Oh 2022, 2023), then train neural networks on them


### Directory Structure:


#### notebooks

#### Primary:
train_CNN_MHD_vs_CRs.ipynb -- Classification of images by turbulence simulation. Can take in either "large" or "small" datasets (depending on whether all slices are included in dataset or just a random subsampling from each of train, validation, and test chunks). Can take in either snapshots of regular gas density or snapshots with power spectra flattened (killPwr flag)

Turb_Unet.ipynb -- Maps from gas density to e.g. magnetic energy density, cosmic ray energy density using a U-net architecture

#### Other:
Turb_StableDiffusion.ipynb -- generates snapshots using stable diffusion (using Hugging Face Diffusers)

Turb_Generation_DDIM -- generates snapshots using DDIM (using Hugging Face Diffusers)

Gaussian_Filter_MHD.ipynb -- Analyzes how a CNN trained to differentiate MHD-only vs MHD+cosmic ray images classifies MHD images that have been gaussian smoothed to varying extents. This smoothing mocks the effects of cosmic rays, which act as a drag force on fluctuations, effectively smoothing out sharp density features.



#### data_creation
fromHDF5_to_fits.py -- converts Athena++ snapshots in HDF5 format to FITS files that are then readable and modifiable with astropy

create_images_multiple_snapshots.py -- most up-to-date version! From input FITS files (../Files/'+fileDir+'/), splits each datacube into 3 chunks, corresponding to train, validation, and test sets, separated by spatial buffers. The buffers are important; without them, gas structures extend from the train set into the validation set, creating correlations between the two. 


Older versions (not recommended):
create_images.py -- from input FITS file in Files/<sim_name> directory, slices the data cube and creates a dataset of ~32,000 images per snapshot. Outputs the *whole* data cube, not split into training, validation, and test sets

create_images_split.py -- same as above but splits data into training, validation, and test sets that are spatially separated so that correlated structures don't occupy two or more sets


#### utilities
load_data.py -- functions that help load pre-split data

interpret_CNN.py -- functions that help interpret the network output, e.g. confusion matrices, saliency maps, occlusion experiments

unet.py -- Pre-made, open-source file from Elektronn3


#### models
Pre-trained models
