# ML_Turb
### These files extract and augment image data from MHD turbulent box simulations (from Bustard and Oh 2023), then train neural networks on them


### Directory Structure:
#### data_creation
fromHDF5_to_fits.py -- converts Athena++ snapshots in HDF5 format to FITS files that are then readable and modifiable with astropy

create_images.py -- from input FITS file in Files/<sim_name> directory, slices the data cube and creates a dataset of ~32,000 images per snapshot. Outputs the *whole* data cube, not split into training, validation, and test sets

create_images_split.py -- same as above but splits data into training, validation, and test sets that are spatially separated so that correlated structures don't occupy two or more sets


#### utilities
load_data.py -- functions that help load pre-split data

interpret_CNN.py -- functions that help interpret the network output, e.g. confusion matrices, saliency maps, occlusion experiments


#### notebooks


#### models
Pre-trained models
