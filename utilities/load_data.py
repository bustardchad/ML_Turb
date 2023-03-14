# Functions that help download, load data files, and convert to DataLoaders
# Input: config
# Output: DataLoaders for training, validation, and test sets
#           i.e. train_dl, valid_dl, test_dl


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset, random_split
import pdb
import os
import gdown

# downloads data folder at URL
def download_data(config):
    # Download the relevant data from Google Drive

    if (config.run_locally==False):

    # Path for the FULL google drive file with turb sim data
    # url = "https://drive.google.com/drive/folders/1C9zPwEglOZI7CqiS4Rz2MESCJzS4wTd2"

    # Path for the smaller turb sim data files (without augmentation)
    #url = "https://drive.google.com/drive/folders/1YDXgeazcwfyciAGUv_sW-gHDy7e1k5wY"

        url = "https://drive.google.com/drive/folders/1zzwYNPSV42jyEQVErlbE-CyKGIustlHv"

        if config.killPwr:
            url = "https://drive.google.com/drive/folders/1B7N_x5Y1N0wH96vKubyQ86Oftaz2M8W9"


        #if not os.path.exists("Image_Cubes_noAugment"):
        gdown.download_folder(url)




# TODO: Put bar labels in correct positions -- doesn't work right now
# Plot histogram showing data distribution by class
def plot_data(classes,fileDirArr):
    plt.figure(figsize=(16,8))
    values, bins, bars = plt.hist(classes)
    #plt.bar_label(bars,labels=fileDirArr, fontsize=20)

    # Set title
    plt.title("Data Distribution",fontsize=22)

    # adding labels
    plt.ylabel('# of images',fontsize = 20)
    plt.show()


def create_data_loaders(config,train_data,val_data,test_data,check_representation=False):
    # load batches of training and validation data
    # the validation data batch size is twice as large because no backprop is needed
    batch_size=config.batch_size

    #load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    valid_dl = DataLoader(val_data, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)
    test_dl = DataLoader(test_data, batch_size*2, shuffle = True, num_workers = 4, pin_memory = True)

    for i, data in enumerate(train_dl):
        if (config.sim_type == 'classify'):
            x, y = data
            imshow(make_grid(x, 8), title = 'Sample batch')
        elif (config.sim_type == 'unet'):
            x, y, z = data
            batch_imshow(make_grid(x, 8), title = 'Sample input batch')
            batch_imshow(make_grid(y, 8), title = 'Sample output batch')
        break  # we need just one batch

    # Optionally check that the training, validation, and test sets have comparable
    # class representation
    if check_representation:
        if (config.sim_type == 'classify'):
            labels_train = []
            for images, labels in train_dl:
                labels_train.extend(labels)

            labels_valid = []
            for images, labels in valid_dl:
                labels_valid.extend(labels)

            labels_test = []
            for images, labels in test_dl:
                labels_test.extend(labels)

        elif (config.sim_type=='unet'):
            labels_train = []
            for images_in, images_out,labels in train_dl:
                labels_train.extend(labels)

            labels_valid = []
            for images_in, images_out, labels in valid_dl:
                labels_valid.extend(labels)

            labels_test = []
            for images_in, images_out, labels in test_dl:
                labels_test.extend(labels)

        plt.hist(labels_train)
        plt.title("training data")
        plt.xlabel("label")
        plt.show()

        plt.hist(labels_valid)
        plt.title("validation data")
        plt.xlabel("label")
        plt.show()

        plt.hist(labels_test)
        plt.title("test data")
        plt.xlabel("label")
        plt.show()


    return train_dl, valid_dl, test_dl




#############################################3
# Code block for classification models
# i.e. config.sim_type = 'classify'

def add_labels(x, x_full, y_full, lbl):
    x = np.float32(x)

    # assign a class value to each image depending on which simulation it came from
    # (e.g. 0 for beta = 1, 1 for beta = 10, 2 for beta = 100)
    y = np.ones(x.shape[2])*lbl # label these images by lbl

    if lbl==0:
        x_full = x
        y_full = y
    else:
        x_full = np.concatenate([x_full,x],axis=2)
        y_full = np.concatenate([y_full,y])

    return x_full, y_full

def reformat(x_full, y_full):
    # x needs to have a channel column, i.e. x will be in format (N,C,H,W)
    # x and y need to have the same first dimension
    x_with_channel = (torch.from_numpy(x_full).permute(2,0,1))
    x_channel_shape = x_with_channel.shape
    x_with_channel = x_with_channel.reshape(-1,1,x_channel_shape[1],x_channel_shape[2])

    y_full = torch.from_numpy(y_full)
    y_full = y_full.type(torch.LongTensor) # throws error unless label is a LongTensor (64)

    return x_with_channel, y_full




# Inputs: fileDirArr -- files.
#         Options are: MHD_beta1, MHD_beta10, MHD_beta100
#              CR_Advect_beta10, CR_Diff_Fiducial_beta10, CR_Diff100_beta10
#         e.g. fileDirArr = ['MHD_beta1','MHD_beta10','MHD_beta100']
#
#         field_list -- list of fields
#         Options are: density, magnetic_energy_density, alfven_speed
#         e.g. field_list = ['density']
def load_presplit_files(config):
    fileDirArr = config.fileDirArr
    field_list = config.field_list
    # For a given field...
    # read in npy files under each file directory
    for field in field_list:
        x_train_full = []
        y_train_full = []
        x_val_full = []
        y_val_full = []
        x_test_full = []
        y_test_full = []
        lbl = 0
        for fileDir in fileDirArr:
            #filename_train = f"/train_{fileDir}_{field}_noAugment.npy"
            filename_train = f"/train_{fileDir}_{field}.npy"
            filename_val = f"/val_{fileDir}_{field}_noAugment.npy"
            filename_test = f"/test_{fileDir}_{field}_noAugment.npy"

            dir = 'Full_Power/'
            if config.killPwr: # use images where power spectra are flattened
                #filename_train = f"/train_{fileDir}_{field}_killPwr_noAugment.npy"
                filename_train = f"/train_{fileDir}_{field}_killPwr.npy"
                filename_val = f"/val_{fileDir}_{field}_killPwr_noAugment.npy"
                filename_test = f"/test_{fileDir}_{field}_killPwr_noAugment.npy"
                dir = 'Kill_Power/'

            x_train = np.load(dir + fileDir + filename_train, mmap_mode='c') # the images
            x_val = np.load(dir + fileDir + filename_val, mmap_mode='c') # the images
            x_test = np.load(dir + fileDir + filename_test, mmap_mode='c') # the images


            x_train_full, y_train_full = add_labels(x_train, x_train_full, y_train_full, lbl)
            x_val_full, y_val_full = add_labels(x_val, x_val_full, y_val_full, lbl)
            x_test_full, y_test_full = add_labels(x_test, x_test_full, y_test_full, lbl)

            lbl+=1

    # bit of reformatting
    img_train_with_channel, labels_train = reformat(x_train_full, y_train_full)
    img_val_with_channel, labels_val = reformat(x_val_full, y_val_full)
    img_test_with_channel, labels_test = reformat(x_test_full, y_test_full)


    if (len(field_list) == 1):
        plot_data(labels_train.numpy(),fileDirArr) # show distribution of data
        plot_data(labels_val.numpy(),fileDirArr) # show distribution of data
        plot_data(labels_test.numpy(),fileDirArr) # show distribution of data

    # Combine into TensorDataset
    train_full = TensorDataset(img_train_with_channel,labels_train)
    val_full = TensorDataset(img_val_with_channel,labels_val)
    test_full = TensorDataset(img_test_with_channel,labels_test)

    return train_full, val_full, test_full



from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# quick plotting function
def imshow(img, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose( img.numpy(), (1, 2, 0)), cmap='gray')
    plt.show()




#########################################
# Code block for U-net mapping between input and target variables
# i.e. config.sim_type = 'unet'

# loading and processing pre-split data

def add_labels_unet(x, x_full, y, y_full, z_full, lbl):
    x = np.float32(x)

    # assign a class value to each image depending on which simulation it came from
    # (e.g. 0 for beta = 1, 1 for beta = 10, 2 for beta = 100)
    z = np.ones(x.shape[2])*lbl # label these images by lbl

    if lbl==0:
        x_full = x
        y_full = y
        z_full = z
    else:
        x_full = np.concatenate([x_full,x],axis=2)
        y_full = np.concatenate([y_full,y],axis=2)
        z_full = np.concatenate([z_full,z])

    return x_full, y_full, z_full

def add_channel(x_full):
    # x needs to have a channel column, i.e. x will be in format (N,C,H,W)
    # x and y need to have the same first dimension
    x_with_channel = (torch.from_numpy(x_full).permute(2,0,1))
    x_channel_shape = x_with_channel.shape
    x_with_channel = x_with_channel.reshape(-1,1,x_channel_shape[1],x_channel_shape[2])

    return x_with_channel

def load_presplit_files_unet(config):
    fileDirArr = config.fileDirArr
    field_list = config.field_list
    # For a given field...
    # read in npy files under each file directory

    field0, field1 = field_list

    x_train_full = []
    y_train_full = []
    z_train_full = []
    x_val_full = []
    y_val_full = []
    z_val_full = []
    x_test_full = []
    y_test_full = []
    z_test_full = []

    lbl = 0

    for fileDir in fileDirArr:
        filename_train0 = f"/train_{fileDir}_{field0}_noAugment.npy"
        filename_val0 = f"/val_{fileDir}_{field0}_noAugment.npy"
        filename_test0 = f"/test_{fileDir}_{field0}_noAugment.npy"

        filename_train1 = f"/train_{fileDir}_{field1}_noAugment.npy"
        filename_val1 = f"/val_{fileDir}_{field1}_noAugment.npy"
        filename_test1 = f"/test_{fileDir}_{field1}_noAugment.npy"

        dir = 'Full_Power/'

        if config.killPwr: # use images where power spectra are flattened
            filename_train0 = f"/train_{fileDir}_{field0}_killPwr_noAugment.npy"
            filename_val0 = f"/val_{fileDir}_{field0}_killPwr_noAugment.npy"
            filename_test0 = f"/test_{fileDir}_{field0}_killPwr_noAugment.npy"

            filename_train1 = f"/train_{fileDir}_{field1}_killPwr_noAugment.npy"
            filename_val1 = f"/val_{fileDir}_{field1}_killPwr_noAugment.npy"
            filename_test1 = f"/test_{fileDir}_{field1}_killPwr_noAugment.npy"

            dir = 'Kill_Power/'

        x_train = np.load(dir + fileDir + filename_train0, mmap_mode='c') # the images
        x_val = np.load(dir + fileDir + filename_val0, mmap_mode='c') # the images
        x_test = np.load(dir + fileDir + filename_test0, mmap_mode='c') # the images

        y_train = np.load(dir + fileDir + filename_train1, mmap_mode='c') # the images
        y_val = np.load(dir + fileDir + filename_val1, mmap_mode='c') # the images
        y_test = np.load(dir + fileDir + filename_test1, mmap_mode='c') # the images

        # assign a class value to each image depending on which simulation it came from
        # (e.g. 0 for beta = 1, 1 for beta = 10, 2 for beta = 100)

        x_train_full, y_train_full, z_train_full = add_labels_unet(x_train,
                                                              x_train_full,
                                                              y_train,
                                                              y_train_full,
                                                              z_train_full,
                                                              lbl)

        x_val_full, y_val_full, z_val_full = add_labels_unet(x_val,
                                                        x_val_full,
                                                        y_val,
                                                        y_val_full,
                                                        z_val_full,
                                                        lbl)

        x_test_full, y_test_full, z_test_full = add_labels_unet(x_test,
                                                        x_test_full,
                                                        y_test,
                                                        y_test_full,
                                                        z_test_full,
                                                        lbl)

        lbl+=1

    # bit of reformatting
    x_train_with_channel = add_channel(x_train_full)
    y_train_with_channel = add_channel(y_train_full)
    x_val_with_channel = add_channel(x_val_full)
    y_val_with_channel = add_channel(y_val_full)
    x_test_with_channel = add_channel(x_test_full)
    y_test_with_channel = add_channel(y_test_full)


    if (len(field_list) == 1):
        #plot_data(labels_train.numpy(),fileDirArr) # show distribution of data
        #plot_data(labels_val.numpy(),fileDirArr) # show distribution of data
        #plot_data(labels_test.numpy(),fileDirArr) # show distribution of data

     # Combine into TensorDataset
     train_full = TensorDataset(x_train_with_channel,y_train_with_channel, z_train_full)
     val_full = TensorDataset(x_val_with_channel, y_val_with_channel, z_val_full)
     test_full = TensorDataset(x_test_with_channel, y_test_with_channel, z_test_full)

     return train_full, val_full, test_full


# loads files assuming they are pre-split into training, validation, and test sets
# returns DataTensors for each split
def preprocess(config):
    train_data, val_data, test_data = load_presplit_files(config)

    # create DataLoaders
    train_dl, valid_dl, test_dl = create_data_loaders(config, train_data,
                                                      val_data, test_data,
                                                      check_representation=True)

    return train_dl, valid_dl, test_dl
















