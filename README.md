# Deep Learning the Effects of Cosmic Rays on Astrophysical Turbulence
### (Chad Bustard and John Wu, in prep)


## Overview
#### The Problem
Cosmic rays are the most highly energetic charged particles in the Universe, and despite representing only ~ a billionth of all particles in the Milky Way galaxy, they exert a sizeable influence on surrounding gas. How they influence gas and what effect they have on the overall evolution of galaxies are very active research topics, but unfortunately, many important outcomes depend sensitively on the (uncertain) microscopic interactions between cosmic ray particles and magnetic waves (think scales of order 10^-6 parsecs rather than galactic scales of order 1000s of parsecs). The predominant way to seek the "ground truth" of this cosmic ray propagation is to compile multiwavelength observations probing direct and indirect detections of cosmic rays and run them through phenomenological models of cosmic ray propagation. Despite monumental efforts on this, we still lack a complete understanding of cosmic ray propagation and, therefore, galaxy evolution as a whole. 

#### The Proposal
As an alternative, we propose that the "ground truth" of cosmic ray propagation might be unearthed by applying machine learning (more specifically, convolutional neural networks) on abundant images of neutral hydrogen, rather than expensive multiwavelength data. Rather than training on observations, though, for which we don't know the ground truth, our proof-of-concept will use simulation data where we have prior knowledge of the cosmic ray propagation. 

#### The Method
In Bustard and Oh 2022 and Bustard Oh 2023, it was shown that cosmic rays, when turbulently stirred with magnetic fields and gas, leave distinct imprints on the gas depending on cosmic ray properties (whether they diffuse or stream along magnetic fields, how fast they diffuse, etc.). Here, we train convolutional neural networks to "learn" these salient imprints and classify cosmic ray properties solely from gas density images. Additionally, to instead of just classify cosmic ray propagation, we also train U-nets to explicitly map gas density images to cosmic ray density images. 

#### The Data
Tens of thousands of images, cautiously split into train, validation, and test sets, are created from the 3D HDF5 data outputs from Bustard and Oh 2023. The training and validation images are predominantly 2D slices of gas density. When we run inference, the test sets are again slices, but we also test images created by averaging over multiple layers (more analogous to true observations). 

#### The Training
We train our networks over 10s of epochs using early stopping and a variety of regularization techniques (mainly dropout) to ward off overfitting. We track classification metrics (e.g. accuracy, precision, recall) and plot confusion matrices to assess model performance, and we iterate over model capacity, learning rate, dropout fraction, etc. during fine-tuning. All training is done on Google Colab using GPUs, hence most scripts in this repo are in Jupyter notebook form. 

#### The Interpretation
We interpret our results via 
1) Saliency maps, which map activations in the CNN to specific regions of images, allowing us to see the salient features that led the CNN to assign an image a certain label.
2) Data manipulation. By training on images with equalized, flattened power spectra, we discover whether any salient *phase* information exists that can help distinguish images, rather than simply salient *spectral* information. We also convolve our input images with Gaussian filters of varying degrees, showing us that image "fuzziness" is a salient, distinguishing property uniquely derived from certain cosmic ray propagation models. 



## Directory Structure:

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

#### notebooks

*Primary*:
train_CNN_MHD_vs_CRs.ipynb -- Classification of images by turbulence simulation. Can take in either "large" or "small" datasets (depending on whether all slices are included in dataset or just a random subsampling from each of train, validation, and test chunks). Can take in either snapshots of regular gas density or snapshots with power spectra flattened (killPwr flag)

Turb_Unet.ipynb -- Maps from gas density to e.g. magnetic energy density, cosmic ray energy density using a U-net architecture

*Other*:
Turb_StableDiffusion.ipynb -- generates snapshots using stable diffusion (using Hugging Face Diffusers)

Turb_Generation_DDIM -- generates snapshots using DDIM (using Hugging Face Diffusers)

Gaussian_Filter_MHD.ipynb -- Analyzes how a CNN trained to differentiate MHD-only vs MHD+cosmic ray images classifies MHD images that have been gaussian smoothed to varying extents. This smoothing mocks the effects of cosmic rays, which act as a drag force on fluctuations, effectively smoothing out sharp density features.



#### models
Pre-trained models
