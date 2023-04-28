# # Methods to help interpret output from the CNN
# #### Requires:
# * Pretrained model and model configuration
#
# #### Interpretability measures considered:
# * Saliency maps
# * Occultation experiments
# * Power spectra and testing CNN on Gaussian filtered images

# ## Import packages

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset, random_split
import pdb
import os

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def confusion(config, model, data_loader, normalize=None):
    # calculate accuracy and return confusion matrix
    #
    # inputs: config file, model, data loader,
    #         normalize = None,'true','pred','all' -- whether to normalize (see sklearn docs)

    classes = config.fileDirArr

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    y_pred = []
    y_true = []

    # again no gradients needed
    with torch.no_grad():
        #for images, labels in train_dl:
        for images, labels in data_loader:  # use validation set here and for saliency maps
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            y_pred.extend(predictions) # save the prediction
            y_true.extend(labels) # save the truth


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


    # Build confusion matrix -- updated to include correct normalization (normalize flag)
    cf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)

    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                         columns = [i for i in classes])
    #df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in classes],
    #                     columns = [i for i in classes])
    fig = plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted Class",fontsize=20)
    plt.ylabel("True Class",fontsize=20)

    return fig


import cmasher as cmr
import random

def saliency(image, model, sigma = 4.0):
    # Requires: image, sigma = standard deviation for gaussian smoothing

    # Steps to make a saliency map
    # 1. Need to load in an image from one of the data sets
    # 2. Evaluate the image with gradients on
    # 3. Run backpropagation and store the gradients
    # 4. Map the gradients to individual pixels
    # 5. Show image
    model.eval() # put in evaluation mode

    # requires a 4D tensor, so need to reshape this 3D one
    image = image.reshape(1, 1, image.shape[1], image.shape[2])


    # we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
    image.requires_grad_()

    # run the model on the image
    outputs = model(image)

    # Get the index corresponding to the maximum score and the maximum score itself.
    pred_max_index = outputs.argmax()
    pred_max = outputs[0,pred_max_index]

    # backward pass to calculate the gradient
    pred_max.backward()

    saliency, _ = torch.max(image.grad[0].data.abs(),dim=0) # dim = 0 is channel?

    # renormalize saliency
    saliency = (saliency - saliency.min())/(saliency.max()-saliency.min())

    # code to plot the saliency map as a heatmap
    plt_image = image.reshape(image.shape[2],image.shape[3])
    #saliency = saliency.reshape(image.shape[2],image.shape[3])
    plt_image = plt_image.detach().numpy()

    # Blur the saliency maps using a gaussian kernel and then overplot contours on the original image
    from scipy.ndimage.filters import gaussian_filter

    filtered_saliency = gaussian_filter(saliency,sigma=sigma)

    return plt_image, filtered_saliency



# custom function to conduct occlusion experiments
# largely taken from:
# https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e

def occlusion(model, image, label, occ_size = 50, occ_stride = 50, occ_pixel = 0.5):
    # Requires: model, an individual image, and a label (class) to return the probability of
    image = image.reshape(1, 1, image.shape[1], image.shape[2])

    #get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    #setting the output image width and height
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))

    #create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    #iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            #run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1) # makes it a probability for each class?
            prob = output.tolist()[0][label]

            #setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap


def plot_saliency(config, model, data_loader, saliency_plot_params):
    # Plots saliency maps for each class, with n_examples per class.
    #
    # Inputs: config, model, data_loader
    #         saliency_plot_params = dictionary of n_examples, n_classes, levels, and figsize

    # Number of examples per class, number of classes
    n_examples = saliency_plot_params['n_examples']
    n_classes = saliency_plot_params['n_classes']

    # Levels = saliency contours with values from 0 to 1, e.g. levels = [0.4, 0.6]
    levels = saliency_plot_params['levels']

    # Size of figure box e.g. (16,12)
    figsize = saliency_plot_params['figsize']

    fig, axs = plt.subplots(n_classes, n_examples, figsize=(4*n_examples,4*n_classes))
    ctr = np.zeros(len(config.fileDirArr))

    # access a batch of labelled images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # select n_example example images from each class
    for i in range(0,n_classes):
        images_to_plot = images[labels==i]
        for j in range(0,n_examples):
            plt_image, filtered_saliency = saliency(images_to_plot[j], model, sigma=4.0)
            # normalize the saliency by its max and plot only one or two levels in the middle range
            filtered_saliency = filtered_saliency/np.max(filtered_saliency)
            axs[i,j].imshow(plt_image,cmap='gray')
            axs[i,j].contour(filtered_saliency,cmap='plasma_r', levels=levels, linewidths = 2)

    cols = ['Example {}'.format(col) for col in range(1, n_examples+1)]
    rows = config.fileDirArr

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')

    fig.tight_layout()

    return fig



def plot_occlusion(config, model, data_loader, n_examples, n_classes, label, occ_size=32, occ_stride=32, occ_pixel=0.5):
    # Plot occlusion results
    n_examples = 6 # number of columns
    n_classes = 2 # number of rows

    fig, axs = plt.subplots(n_classes*2, n_examples, figsize=(16,8))
    ctr = np.zeros(len(config.fileDirArr))

    # access a batch of labelled images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # select n_example example images from each class
    for i in range(0,n_classes):
        images_to_plot = images[labels==i]
        for j in range(0,n_examples):
            plt_image = images_to_plot[j,:,:,:]

            probs = occlusion(model, plt_image, label, occ_size, occ_stride, occ_pixel)

            plt_image = plt_image.squeeze()

            axs[2*i,j].imshow(plt_image,cmap='gray')

            heatmap = axs[2*i+1,j].imshow(probs, cmap='Reds_r',vmin = probs.min(), vmax = probs.max())

            color_bar = fig.colorbar(heatmap,
                         ax = axs[2*i+1,j],
                         extend = 'both')


    cols = ['Example {}'.format(col) for col in range(1, n_examples+1)]
    rows = config.fileDirArr

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')

    fig.tight_layout()

    return fig



import scipy.stats as stats

def power1D(image):
    # Function to create 1D power spectrum from image

    image = image.reshape(image.shape[-2],image.shape[-1])
    npix = image.shape[1]
    power = 1.0

    fourier_image = np.fft.fftn(image**power)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)

    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    k = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return k, Abins






# Grad-cam section -- still very much under construction
"""
!pip install grad-cam


# taken straight from this link for now: https://jacobgil.github.io/pytorch-gradcam-book/introduction.html

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2

#print(model.children)
#target
#target_layers = [model.children()[0]]
# how do you access layers from nn.Sequential??

target_layers = [model[-4]]

dataiter = iter(train_dl)
images, labels = next(dataiter)
img = images[0,:,:,:].numpy()
gray_img_float = np.float32(img) / 255

input_tensor = torch.from_numpy(gray_img_float).reshape(1,1,128,128)

# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers = target_layers, use_cuda=False)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

#targets = [ClassifierOutputTarget(1)]
targets = None

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
#grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(gray_img_float, grayscale_cam[0,:], use_rgb=True)



from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
# Create the metric target, often the confidence drop in a score of some category
metric_target = ClassifierOutputSoftmaxTarget(281)
scores, batch_visualizations = CamMultImageConfidenceChange()(input_tensor,
  inverse_cams, targets, model, return_visualization=True)
visualization = deprocess_image(batch_visualizations[0, :])

# State of the art metric: Remove and Debias
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst
cam_metric = ROADMostRelevantFirst(percentile=75)
scores, perturbation_visualizations = cam_metric(input_tensor,
  grayscale_cams, targets, model, return_visualization=True)

# You can also average accross different percentiles, and combine
# (LeastRelevantFirst - MostRelevantFirst) / 2
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage,
                                          ROADLeastRelevantFirstAverage,
                                          ROADCombined
cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
scores = cam_metric(input_tensor, grayscale_cams, targets, model)
"""
