# ML_Turb
### These files extract and augment image data from MHD turbulent box simulations (from Bustard and Oh 2023), then train neural networks on them


#### Files
fromHDF5_to_fits.py -- converts Athena++ snapshots in HDF5 format to FITS files that are then readable and modifiable with astropy

create_images.py -- from input FITS file in Files/<sim_name> directory, slices the data cube and creates a dataset of ~32,000 images per snapshot

train_cnn.ipynb -- main notebook, uses PyTorch to create and train a CNN in order to classify images by their simulation class: { $\beta \sim 1$, $\beta \sim 10$, $\beta \sim 100$}
