# Helper functions for U-net models

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchmetrics import StructuralSimilarityIndexMeasure

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up loss function and prediction function with metrics to track
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

def ssim_loss(x, y, alpha):
    """
    Calculate the SSIM loss between two images.

    Args:
        x (torch.Tensor): Prediction.
        y (torch.Tensor): Target.
        alpha (float): Weighting factor for SSIM loss.

    Returns:
        torch.Tensor: SSIM loss.
    """
    return alpha * (1.0 - ssim(x, y))

def _early_stopping(config, model, val_loss):
    """
    Implement early stopping mechanism.

    Args:
        config: Configuration object.
        model: PyTorch model.
        val_loss (list): List of validation losses.

    Returns:
        bool: Whether to continue training or not.
    """
    if len(val_loss) > 1:
        if val_loss[-1] < val_loss[-2]:
            print("Saving model checkpoint")
            torch.save(model.state_dict(), config.save_dir + config.run_name + '_checkpoint.pt')
            return True
    if len(val_loss) > config.patience:
        if (np.diff(val_loss[-config.patience:]) <= 0).all():
            print("Early Stopping")
            return False

def loss_batch(config, loss_func, prediction, yb, opt=None, struc_alpha=0.2):
    """
    Calculate loss for a batch.

    Args:
        config: Configuration object.
        loss_func: Loss function.
        prediction (torch.Tensor): Model's prediction.
        yb (torch.Tensor): Target.
        opt: Optimizer.
        struc_alpha (float): Weighting factor for structural loss.

    Returns:
        tuple: Tuple containing loss, structural loss, and batch size.
    """
    loss = loss_func(prediction, yb)
    if config.use_ssim:
        struc_loss = ssim_loss(prediction, yb, struc_alpha)
        loss += struc_loss
    else:
        struc_loss = 0

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), struc_loss.item(), len(yb)

def fit(config, model, loss_func, opt, train_dl, valid_dl, alpha=0.2):
    """
    Train the U-Net model.

    Args:
        config: Configuration object.
        model: PyTorch model.
        loss_func: Loss function.
        opt: Optimizer.
        train_dl: Training data loader.
        valid_dl: Validation data loader.
        alpha (float): Weighting factor for structural loss.

    Returns:
        tuple: Model, training loss array, validation loss array,
            training structural similarity array, validation structural similarity array.
    """
    if config.use_ssim:
        df = pd.DataFrame(columns=["Epoch", "Train_Loss", "Validation_Loss", "Train_SSIM", "Validation_SSIM"])
    else:
        df = pd.DataFrame(columns=["Epoch", "Train_Loss", "Validation_Loss"])

    loss_arr_train = []
    loss_arr_val = []
    struc_arr_train = []
    struc_arr_val = []
    keep_training = True

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_struc = 0
        num = 0

        for xb, yb, lbl in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            prediction = model(xb)

            loss, struc_loss, len = loss_batch(config, loss_func, prediction, yb, opt, alpha)

            if config.use_ssim:
                train_struc += -(struc_loss / alpha - 1.0) * len

            train_loss += (loss * len)
            num += len

        train_loss = train_loss / num
        if config.use_ssim:
            train_struc = train_struc / num

        model.eval()

        with torch.no_grad():
            val_loss = 0
            val_struc = 0
            num = 0

            for xb, yb, lbl in valid_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                prediction = model(xb)

                loss, struc_loss, len = loss_batch(config, loss_func, prediction, yb, None, alpha)

                if config.use_ssim:
                    val_struc += -(struc_loss / alpha - 1.0) * len

                val_loss += (loss * len)
                num += len

        if config.use_ssim:
            val_struc = val_struc / num

        val_loss = val_loss / num

        loss_arr_train.append(train_loss)
        loss_arr_val.append(val_loss)

        df2 = {'Epoch': epoch, 'Train_Loss': train_loss, 'Validation_Loss': val_loss}

        if config.use_ssim:
            struc_arr_train.append(train_struc)
            struc_arr_val.append(val_struc)
            df2 = {'Epoch': epoch, 'Train_Loss': train_loss, 'Validation_Loss': val_loss,
                   'Train_SSIM': train_struc, 'Validation_SSIM': val_struc}

        df = df.append(df2, ignore_index=True)

        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}| Validation loss: {val_loss:.5f}")

            filename = config.save_dir + "Loss_" + config.run_name + ".csv"
            df.to_csv(filename)

            model_for_eval = model
            plot_results(config, model_for_eval.cpu(), valid_dl, epoch)
            model = model.to(device)

        keep_training = _early_stopping(config, model, loss_arr_val)

        if keep_training == False and config.stop_early == True:
            print("Early Stopping")
            model.load_state_dict(torch.load(config.save_dir + config.run_name + '_checkpoint.pt'))
            break

    return model, loss_arr_train, loss_arr_val, struc_arr_train, struc_arr_val

def plot_results(config, model_eval, data_loader, epoch):
    """
    Plot the results of image batches after U-Net mapping.

    Args:
        config: Configuration object.
        model_eval: PyTorch model for evaluation.
        data_loader: Data loader for images.
        epoch (int): Current epoch.
    """
    model_eval.eval()

    dataiter = iter(data_loader)
    x_arr, y_arr, labels = next(dataiter)

    n_examples = 6
    fig, axs = plt.subplots(3, n_examples, figsize=(int(8 * n_examples / 3), 8))
    with plt.style.context('fast'):
        for i in range(0, n_examples):
            x = x_arr[i]
            y = y_arr[i]
            image = x.reshape(1, 1, x.shape[1], x.shape[2])
            target = y.reshape(1, 1, y.shape[1], y.shape[2])

            target = target.reshape(target.shape[2], target.shape[3])
            target = target.detach().numpy()

            image.requires_grad_()

            outputs = model_eval(image)

            input_image = image.reshape(image.shape[2], image.shape[3])
            input_image = input_image.detach().numpy()

            target_image = outputs.reshape(image.shape[2], image.shape[3])
            target_image = target_image.detach().numpy()

            axs[0, i].imshow(input_image, cmap='gray')
            axs[1, i].imshow(target, cmap='gray')
            axs[2, i].imshow(target_image, cmap='gray')

        cols = ['Example {}'.format(col) for col in range(1, n_examples + 1)]
        rows = ['Input', 'Real Target', 'Modeled Target']

        for ax, col in zip(axs[0], cols):
            ax.set_title(col)

        for ax, row in zip(axs[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='large')

        fig.tight_layout()
        plt.savefig(config.save_dir + config.run_name + '_image_epoch' + str(epoch) + '.png', dpi=600)
        plt.show()
        plt.close()

from torchvision.utils import make_grid

def batch_imshow(img, title):
    """
    Show a batch of images.

    Args:
        img: Image batch.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)), cmap='gray')
    plt.show()

