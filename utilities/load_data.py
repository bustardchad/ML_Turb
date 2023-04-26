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
#          dataset_size = 'large' or 'small' (large includes 4x more images)
#          path_to_dir = '../'
#               -- set to '' if downloading files from Google Drive
#               -- set to '../Full_Power/' or '../Kill_Power/' if running on local computer
#               -- set to full path to fileDirArr if mounting Google Drive on Colab
#
# Output: DataLoaders for training, validation, and test sets
#           i.e. train_dl, valid_dl, test_dl


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, ConcatDataset, TensorDataset, random_split
import torchvision.transforms as T
import pdb
import os
import gdown


class CustomUnetDataset(Dataset):
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

        if ((not np.any(x.numpy() > 0) or (not np.any(y.numpy() > 0))): # skip the images that are blank
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


class CustomClassifyDataset(Dataset):
    """TensorDataset that supports transforms for input imaas
       Assumes there are only input images (x) with labels (y)
    """
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.inputs[0][index]
        y = self.labels[index]

        if not np.any(x.numpy() > 0): # skip the images that are blank
            rep_index = np.random.randint(0, 8)
            return self.__getitem__(rep_index)

        if self.transform:
            x = self.transform(x)


        return x, y

    def __len__(self):
        return self.inputs[0].size(0)




# downloads data folder at URL dependent upon config settings
def download_data(config):
    # Download the relevant data from Google Drive

    if (config.run_locally==False):

        if ((config.dataset_size == 'small') and (config.killPwr == False)):
            url = "https://drive.google.com/drive/folders/1gSfyMstWIO8BjAoD_6t-a5xDmMd6cfCW"
        elif ((config.dataset_size == 'large') and (config.killPwr == False)):
            raise Exception("Large dataset not loaded to Google Drive yet")
        elif ((config.dataset_size == 'small') and (config.killPwr == True)):
            url = "https://drive.google.com/drive/folders/1_QCS78V1XB7YYTQc2alutmC1wmNSt9nx"
        elif ((config.dataset_size == 'large') and (config.killPwr == True)):
            raise Exception("Large dataset not loaded to Google Drive yet")
        else:
            raise Warning("Invalid dataset_size or killPwr flag, will default to 'small' and 'false'")


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
    if (config.hold_out_test_set == True):
        train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        test_dl = DataLoader(test_data, batch_size*2, shuffle = True, num_workers = 0, pin_memory = True)
    else:
        train_dl = DataLoader(ConcatDataset([train_data,test_data]), batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        test_dl = DataLoader(test_data, batch_size*2, shuffle = True, num_workers = 0, pin_memory = True)

    valid_dl = DataLoader(val_data, batch_size*2, shuffle = True, num_workers = 0, pin_memory = True)

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


def add_transforms(config, tensors, labels):
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


    # Combine into Dataset
    # Type of CustomDataset is determined based on sim_type

    if config.sim_type=='classify':
        no_transforms = CustomClassifyDataset(tensors,labels,transform=None)

        if config.use_transforms:
            horiz_transforms = CustomClassifyDataset(tensors,labels,transform=horiz)
            vert_transforms = CustomClassifyDataset(tensors,labels,transform=vert)

            # concatenate the non-transformed and transformed datasets
            full_dataset = torch.utils.data.ConcatDataset([no_transforms, horiz_transforms, vert_transforms])
        else:
            full_dataset = no_transforms

    elif config.sim_type=='unet':
        no_transforms = CustomUnetDataset(tensors,labels,transform=None)

        if config.use_transforms:
            horiz_transforms = CustomUnetDataset(tensors,labels,transform=horiz)
            vert_transforms = CustomUnetDataset(tensors,labels,transform=vert)

            # concatenate the non-transformed and transformed datasets
            full_dataset = torch.utils.data.ConcatDataset([no_transforms, horiz_transforms, vert_transforms])
        else:
            full_dataset = no_transforms

    else:
        raise Exception("sim_type is not valid, must be either 'classify' or 'unet' ")

    return full_dataset


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
            if config.dataset_size=='large':
                filename_train = f"/train_{fileDir}_{field}_large.npy"
                filename_val = f"/val_{fileDir}_{field}_large.npy"
                filename_test = f"/test_{fileDir}_{field}_large.npy"
            else:
                filename_train = f"/train_{fileDir}_{field}_small.npy"
                filename_val = f"/val_{fileDir}_{field}_small.npy"
                filename_test = f"/test_{fileDir}_{field}_small.npy"

            dir = config.path_to_dir

            if config.killPwr: # use images where power spectra are flattened
                if config.dataset_size=='large':
                    filename_train = f"/train_{fileDir}_{field}_large.npy"
                    filename_val = f"/val_{fileDir}_{field}_large.npy"
                    filename_test = f"/test_{fileDir}_{field}_large.npy"
                else:
                    filename_train = f"/train_{fileDir}_{field}_small.npy"
                    filename_val = f"/val_{fileDir}_{field}_small.npy"
                    filename_test = f"/test_{fileDir}_{field}_small.npy"

                dir = config.path_to_dir

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

    # Add transforms (if wanted) and return as Datasets
    train_full = add_transforms(config, [img_train_with_channel], labels_train)
    val_full = add_transforms(config, [img_val_with_channel], labels_val)
    test_full = add_transforms(config, [img_test_with_channel], labels_test)

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

    # if files are from first class, make x_full = x, etc.
    if lbl==0:
        x_full = x
        y_full = y
        z_full = z
    else: # add new x, y, z to pre-existing x_full, y_full, z_full
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

    # path to files in fileDirArr
    dir = config.path_to_dir

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
        filename_train0 = f"/train_{fileDir}_{field0}_"+str(config.dataset_size)+".npy"
        filename_val0 = f"/val_{fileDir}_{field0}_"+str(config.dataset_size)+".npy"
        filename_test0 = f"/test_{fileDir}_{field0}_"+str(config.dataset_size)+".npy"

        filename_train1 = f"/train_{fileDir}_{field1}_"+str(config.dataset_size)+".npy"
        filename_val1 = f"/val_{fileDir}_{field1}_"+str(config.dataset_size)+".npy"
        filename_test1 = f"/test_{fileDir}_{field1}_"+str(config.dataset_size)+".npy"


        x_train = np.load(dir + fileDir + filename_train0, mmap_mode='c') # the images
        x_val = np.load(dir + fileDir + filename_val0, mmap_mode='c') # the images
        x_test = np.load(dir + fileDir + filename_test0, mmap_mode='c') # the images

        y_train = np.load(dir + fileDir + filename_train1, mmap_mode='c') # the images
        y_val = np.load(dir + fileDir + filename_val1, mmap_mode='c') # the images
        y_test = np.load(dir + fileDir + filename_test1, mmap_mode='c') # the images

        # assign a class value to each image depending on which simulation it came from
        # (e.g. 0 for beta = 1, 1 for beta = 10, 2 for beta = 100)

        # updates x_train_full, y_train_full, z_train_full with data from each fileDir
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

    #print("Debugging: ")
    #print("Shape of data before making into datasets and adding transforms: ")
    #print(x_train_with_channel.shape, y_train_with_channel.shape, z_train_full.shape)

    # add transforms (if wanted) and return as Datasets
    train_full = add_transforms(config, [x_train_with_channel, y_train_with_channel], z_train_full)
    val_full = add_transforms(config, [x_val_with_channel, y_val_with_channel], z_val_full)
    test_full = add_transforms(config, [x_test_with_channel, y_test_with_channel], z_test_full)

    return train_full, val_full, test_full


# loads files assuming they are pre-split into training, validation, and test sets
# returns DataTensors for each split
def preprocess(config):
    if (config.sim_type=='classify'):
        train_data, val_data, test_data = load_presplit_files(config)

        # create DataLoaders
        train_dl, valid_dl, test_dl = create_data_loaders(config, train_data,
                                                      val_data, test_data,
                                                      check_representation=True)
    elif (config.sim_type=='unet'):
        train_data, val_data, test_data = load_presplit_files_unet(config)

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
        fileDirArr = ['CR_Advect_beta10','CR_Diff_Fiducial_beta10']
        field_list = ['density','Ec']
        data_presplit = True # whether data has already been split into training, val, test
        killPwr = False
        run_locally = True
        run_colab = False
        use_transforms = False
        dataset_size = 'small'
        hold_out_test_set = True
        path_to_dir = '../Full_Power/'

    config_test = TestConfig()


    # TODO: Make this a series of assertions


    print("######################################")
    # call preprocess with and without transforms
    print("Loading files for U-net without transformations")
    print("...")
    print("...")
    print("...")
    print("...")
    train_dl, valid_dl, test_dl = preprocess(config_test)

    # print sizes of datasets
    print("Size of training data: ")
    print(len(train_dl.dataset))
    print("...")
    print("...")
    print("...")
    print("...")

    ####################
    config_test.hold_out_test_set = True
    # call preprocess with and without transforms
    print("Loading files for U-net with test set held out of training set")
    print("...")
    print("...")
    print("...")
    print("...")
    train_dl, valid_dl, test_dl = preprocess(config_test)

    # print sizes of datasets
    print("Size of training data: ")
    print(len(train_dl.dataset))
    print("...")
    print("...")
    print("...")

    #########################3
    config_test.use_transforms = True
    print("Loading files for U-net WITH transformations")
    print("...")
    print("...")
    print("...")
    print("...")
    train_dl, valid_dl, test_dl = preprocess(config_test)

    # print sizes of datasets
    print("Size of training data: ")
    print(len(train_dl.dataset))
    print("...")
    print("...")
    print("...")
    print("...")

    print("######################################")

    config_test.sim_type = 'classify'
    config_test.use_transforms = False

    # call preprocess with and without transforms
    print("Loading files for classification without transformations")
    print("...")
    print("...")
    print("...")
    print("...")
    train_dl, valid_dl, test_dl = preprocess(config_test)

    # print sizes of datasets
    print("Size of training data: ")
    print(len(train_dl.dataset))
    print("...")
    print("...")
    print("...")
    print("...")

    config_test.use_transforms = True
    print("Loading files for classification WITH transformations")
    print("...")
    print("...")
    print("...")
    print("...")
    train_dl, valid_dl, test_dl = preprocess(config_test)

    # print sizes of datasets
    print("Size of training data: ")
    print(len(train_dl.dataset))


    """
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
    """
















