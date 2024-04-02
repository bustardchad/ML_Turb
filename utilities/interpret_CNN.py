# Helper functions to interpret CNN via saliency maps, occlusion experiments, etc.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset, random_split
import os
import pdb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import cmasher as cmr
import random
import scipy.stats as stats

def confusion(config, model, data_loader, confusion_plot_params=None, normalize=None):
    """
    Calculate accuracy and return a confusion matrix.

    Args:
        config: Configuration object.
        model: PyTorch model.
        data_loader: DataLoader.
        confusion_plot_params (dict): Parameters for confusion plot. Defaults to None.
        normalize (str): Normalize option for confusion matrix. Defaults to None.

    Returns:
        plt.figure: Confusion matrix figure.
    """
    if confusion_plot_params['classnames']:
        classes = confusion_plot_params['classnames']
    else:
        classes = config.fileDirArr

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            y_pred.extend(predictions)
            y_true.extend(labels)

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    cf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    fig = plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted Class", fontsize=20)
    plt.ylabel("True Class", fontsize=20)
    fig.tight_layout()
    return fig

def saliency(image, model, sigma=4.0):
    """
    Compute saliency map for an image.

    Args:
        image: Input image.
        model: PyTorch model.
        sigma (float): Standard deviation for Gaussian smoothing. Defaults to 4.0.

    Returns:
        tuple: Processed image and filtered saliency map.
    """
    model.eval()
    image = image.reshape(1, 1, image.shape[1], image.shape[2])
    image.requires_grad_()
    outputs = model(image)
    pred_max_index = outputs.argmax()
    pred_max = outputs[0, pred_max_index]
    pred_max.backward()
    saliency, _ = torch.max(image.grad[0].data.abs(), dim=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    plt_image = image.reshape(image.shape[2], image.shape[3]).detach().numpy()
    filtered_saliency = gaussian_filter(saliency, sigma=sigma)
    return plt_image, filtered_saliency

def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    """
    Perform occlusion experiment.

    Args:
        model: PyTorch model.
        image: Input image.
        label: Class label.
        occ_size (int): Occlusion size. Defaults to 50.
        occ_stride (int): Occlusion stride. Defaults to 50.
        occ_pixel (float): Occlusion pixel value. Defaults to 0.5.

    Returns:
        torch.Tensor: Heatmap
    """
    image = image.reshape(1, 1, image.shape[1], image.shape[2])
    width, height = image.shape[-2], image.shape[-1]
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))
    heatmap = torch.zeros((output_height, output_width))

    for h in range(0, height):
        for w in range(0, width):
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            if (w_end) >= width or (h_end) >= height:
                continue
            input_image = image.clone().detach()
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]
            heatmap[h, w] = prob

    return heatmap

def plot_saliency(config, model, data_loader, saliency_plot_params):
    """
    Plot saliency maps.

    Args:
        config: Configuration object.
        model: PyTorch model.
        data_loader: DataLoader.
        saliency_plot_params (dict): Parameters for saliency plot.

    Returns:
        plt.figure: Salience plot figure.
    """
    n_examples = saliency_plot_params['n_examples']
    n_classes = saliency_plot_params['n_classes']
    levels = saliency_plot_params['levels']
    figsize = saliency_plot_params['figsize']

    fig, axs = plt.subplots(n_classes, n_examples, figsize=(4 * n_examples, 4 * n_classes))
    ctr = np.zeros(len(config.fileDirArr))
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    for i in range(0, n_classes):
        images_to_plot = images[labels == i]
        for j in range(0, n_examples):
            plt_image, filtered_saliency = saliency(images_to_plot[j], model, sigma=4.0)
            filtered_saliency = filtered_saliency / np.max(filtered_saliency)
            axs[i, j].imshow(plt_image, cmap='gray')
            axs[i, j].contour(filtered_saliency, cmap='plasma_r', levels=levels, linewidths=2)

    cols = ['Example {}'.format(col) for col in range(1, n_examples + 1)]
    if saliency_plot_params['classnames']:
        rows = saliency_plot_params['classnames']
    else:
        rows = config.fileDirArr

    for ax, col in zip(axs[0], cols):
        ax.set_title(col, color="k", pad=6.0, size='x-large', fontweight='semibold')

    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=90, size='xx-large', fontweight='semibold')

    fig.tight_layout()
    return fig

def power1D(image):
    """
    Create 1D power spectrum from image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        tuple: k values and power spectrum.
    """
    image = image.reshape(image.shape[-2], image.shape[-1])
    npix = image.shape[1]
    power = 1.0

    fourier_image = np.fft.fftn(image ** power)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)

    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    k = 0.5 * (kbins[1:] + kbins[:-1])

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    return k, Abins


