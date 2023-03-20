# Functions that help download, load data files, and convert to DataLoaders
#
# Input:  config with the following parameters:
#          sim_type = 'unet' or 'classify'
#          batch_size = 64
#          fileDirArr = ['MHD_beta10'] -- file directories
#          field_list = ['density','magnetic_energy_density'] -- fields to load
#          data_presplit = True -- flag for whether data has already been split into training, val, test
#          killPwr = False -- whether we should load data with power spectra flattened or not
#          run_locally = True -- only set to True if running on my local computer
#          run_colab = False -- set to true if running on Google Colab
#          use_transforms = False -- if True, augment data with horizontal and vertical flips
#          path_to_dir = '../' -- should be set to '' if Full_Power or Kill_Power directories are in same folder as this script
#
# Output: DataLoaders for training, validation, and test sets
#           i.e. train_dl, valid_dl, test_dl


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, TensorDataset, random_split
import torchvision.transforms as T
import pdb
import os
import gdown


class CustomDataset(Dataset):
    """TensorDataset that supports transforms for both input and target images
    """
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.inputs[0][index]
        y = self.inputs[1][index]
        z = self.labels[index]

        if not np.any(x.numpy() > 0): # skip the images that are blank
            rep_index = np.random.randint(0, 8)
            return self.__getitem__(rep_index)
        #assert np.any(x.numpy() > 0), "Error: input image is blank"
        #assert np.any(y.numpy() > 0), "Error: target image is blank"

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)


        return x, y, z

    def __len__(self):
        return self.inputs[0].size(0)



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


from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def batch_imshow(img, title):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose( img.detach().cpu().numpy(), (1, 2, 0)), cmap='gray')
    plt.show()

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
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
    valid_dl = DataLoader(val_data, batch_size*2, shuffle = True, num_workers = 0, pin_memory = True)
    test_dl = DataLoader(test_data, batch_size*2, shuffle = True, num_workers = 0, pin_memory = True)

    """
    for i, data in enumerate(train_dl):
        if (config.sim_type == 'classify'):
            x, y = data
            imshow(make_grid(x, 8), title = 'Sample batch')
        elif (config.sim_type == 'unet'):
            print("Unet")
            x, y, z = data
            print(x.shape)
            print(y.shape)
            print(z.shape)
            batch_imshow(make_grid(x, 8), title = 'Sample input batch')
            batch_imshow(make_grid(y, 8), title = 'Sample output batch')
        break  # we need just one batch
    """
    # Optionally check that the training, validation, and test sets have comparable
    # class representation
    """
    if check_representation:
        labels_train = []
        labels_valid = []
        labels_test = []

        if (config.sim_type == 'classify'):
            for images, labels in train_dl:
                labels_train.extend(labels)

            for images, labels in valid_dl:
                labels_valid.extend(labels)

            for images, labels in test_dl:
                labels_test.extend(labels)

        elif (config.sim_type=='unet'):
            for data, labels in train_dl:
                labels_train.extend(labels)

            for data, labels in valid_dl:
                labels_valid.extend(labels)

            for data, labels in test_dl:
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
    """

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


def add_transforms(tensors,labels):
    horiz = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(1.0),
            T.ToTensor()
            ])

    vert = T.Compose([
            T.ToPILImage(),
            T.RandomVerticalFlip(1.0),
            T.ToTensor()
            ])


    # Combine into TensorDataset
    no_transforms = CustomDataset(tensors,labels,transform=None)
    horiz_transforms = CustomDataset(tensors,labels,transform=horiz)
    vert_transforms = CustomDataset(tensors,labels,transform=vert)

    full_dataset = torch.utils.data.ConcatDataset([no_transforms, horiz_transforms, vert_transforms])

    return full_dataset


# Inputs: fileDirArr -- files.
#         Options are: MHD_beta1, MHD_beta10, MHD_beta100
#              CR_Advect_beta10, CR_Diff_Fiducial_beta10, CR_Diff100_beta10
#         e.g. fileDirArr = ['MHD_beta1','MHD_beta10','MHD_beta100']
#
#         field_list -- list of fields
#         Options are: density, magnetic_energy_density, alfven_speed
#         e.g. field_list = ['density']
def load_presplit_files(config,augment=False):
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
            if augment:
                filename_train = f"/train_{fileDir}_{field}.npy"
            else:
                filename_train = f"/train_{fileDir}_{field}_noAugment.npy"

            filename_val = f"/val_{fileDir}_{field}_noAugment.npy"
            filename_test = f"/test_{fileDir}_{field}_noAugment.npy"

            dir = config.path_to_dir + 'Full_Power/'

            if config.killPwr: # use images where power spectra are flattened
                #filename_train = f"/train_{fileDir}_{field}_killPwr_noAugment.npy"
                if augment:
                    filename_train = f"/train_{fileDir}_{field}_killPwr.npy"
                else:
                    filename_train = f"/train_{fileDir}_{field}_killPwr_noAugment.npy"

                filename_val = f"/val_{fileDir}_{field}_killPwr_noAugment.npy"
                filename_test = f"/test_{fileDir}_{field}_killPwr_noAugment.npy"

                dir = config.path_to_dir + 'Kill_Power/'

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

    # TODO: Check this and make sure it transforms the target images too when doing a U-net...
    if config.use_transforms:
        train_full = add_transforms([img_train_with_channel], labels_train)
        val_full = add_transforms([img_val_with_channel], labels_val)
        test_full = add_transforms([img_test_with_channel], labels_test)
    else:
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
    y = np.float32(y)

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

def load_presplit_files_unet(config,augment=False):
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
        if augment:
            filename_train0 = f"/train_{fileDir}_{field0}.npy"
        else:
            filename_train0 = f"/train_{fileDir}_{field0}_noAugment.npy"

        filename_val0 = f"/val_{fileDir}_{field0}_noAugment.npy"
        filename_test0 = f"/test_{fileDir}_{field0}_noAugment.npy"

        if augment:
            filename_train1 = f"/train_{fileDir}_{field1}.npy"
        else:
            filename_train1 = f"/train_{fileDir}_{field1}_noAugment.npy"

        filename_val1 = f"/val_{fileDir}_{field1}_noAugment.npy"
        filename_test1 = f"/test_{fileDir}_{field1}_noAugment.npy"

        dir = config.path_to_dir + 'Full_Power/'


        if config.killPwr: # use images where power spectra are flattened
            if augment:
                filename_train0 = f"/train_{fileDir}_{field0}_killPwr.npy"
            else:
                filename_train0 = f"/train_{fileDir}_{field0}_killPwr_noAugment.npy"

            filename_val0 = f"/val_{fileDir}_{field0}_killPwr_noAugment.npy"
            filename_test0 = f"/test_{fileDir}_{field0}_killPwr_noAugment.npy"

            if augment:
                filename_train1 = f"/train_{fileDir}_{field1}_killPwr.npy"
            else:
                filename_train1 = f"/train_{fileDir}_{field1}_killPwr_noAugment.npy"

            filename_val1 = f"/val_{fileDir}_{field1}_killPwr_noAugment.npy"
            filename_test1 = f"/test_{fileDir}_{field1}_killPwr_noAugment.npy"

            dir = config.path_to_dir + 'Kill_Power/'

        print("Training filenames:")
        print(filename_train0)
        print(filename_train1)

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

    z_train_full = torch.from_numpy(z_train_full)
    z_train_full = z_train_full.type(torch.LongTensor) # throws error unless label is a LongTensor (64)
    z_val_full = torch.from_numpy(z_val_full)
    z_val_full = z_val_full.type(torch.LongTensor) # throws error unless label is a LongTensor (64)
    z_test_full = torch.from_numpy(z_test_full)
    z_test_full = z_test_full.type(torch.LongTensor) # throws error unless label is a LongTensor (64)

    if config.use_transforms:
        train_full = add_transforms([x_train_with_channel, y_train_with_channel], z_train_full)
        val_full = add_transforms([x_val_with_channel, y_val_with_channel], z_val_full)
        test_full = add_transforms([x_test_with_channel, y_test_with_channel], z_test_full)

    else:
        train_full = CustomDataset([x_train_with_channel,y_train_with_channel], z_train_full)
        val_full = CustomDataset([x_val_with_channel, y_val_with_channel], z_val_full)
        test_full = CustomDataset([x_test_with_channel, y_test_with_channel], z_test_full)

    return train_full, val_full, test_full


# loads files assuming they are pre-split into training, validation, and test sets
# returns DataTensors for each split
def preprocess(config,augment=False):
    if (config.sim_type=='classify'):
        train_data, val_data, test_data = load_presplit_files(config,augment)

        # create DataLoaders
        train_dl, valid_dl, test_dl = create_data_loaders(config, train_data,
                                                      val_data, test_data,
                                                      check_representation=True)
    elif (config.sim_type=='unet'):
        train_data, val_data, test_data = load_presplit_files_unet(config,augment)

        # create DataLoaders
        train_dl, valid_dl, test_dl = create_data_loaders(config, train_data,
                                                      val_data, test_data,
                                                      check_representation=False)
    return train_dl, valid_dl, test_dl



# For unit testing #
from dataclasses import dataclass

def _test_loader_():

    @dataclass
    class TestConfig:
        sim_type = 'unet'
        batch_size = 64
        fileDirArr = ['MHD_beta10']
        field_list = ['density','magnetic_energy_density']
        data_presplit = True # whether data has already been split into training, val, test
        killPwr = False
        run_locally = True
        run_colab = False
        use_transforms = False
        path_to_dir = '../'

    config_test = TestConfig()

    # other inputs needed
    augment = True

    # call preprocess(config, augment) with and without transforms
    train_dl, valid_dl, test_dl = preprocess(config_test,augment)

    # print sizes of datasets
    print("Size of training data: ")
    print(train_dl.__len__())

    for i, data in enumerate(train_dl):
        x, y, z = data
        batch_imshow(make_grid(x, 8), title = 'Training: Sample density batch')
        batch_imshow(make_grid(y, 8), title = 'Training: Sample magnetic energy density batch')
        break  # we need just one batch

    for i, data in enumerate(valid_dl):
        x, y, z = data
        batch_imshow(make_grid(x, 8), title = 'Validation: Sample density batch')
        batch_imshow(make_grid(y, 8), title = 'Validation: Sample magnetic energy density batch')
        break  # we need just one batch

    for i, data in enumerate(test_dl):
        x, y, z = data
        batch_imshow(make_grid(x, 8), title = 'Testing: Sample density batch')
        batch_imshow(make_grid(y, 8), title = 'Testing: Sample magnetic energy density batch')
        break  # we need just one batch

















