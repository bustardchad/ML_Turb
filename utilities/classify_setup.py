# Helper functions for setting up the classification network

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


def _early_stopping(config, model, val_loss):
    """
    Check if validation loss has decreased or increased for a certain number of epochs.

    Args:
        config: Configuration object.
        model: PyTorch model.
        val_loss: List of validation losses.

    Returns:
        bool: True if continue training, False if early stopping criterion met.
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


def loss_batch(loss_func, prediction, yb, opt=None):
    """
    Compute loss for a batch and perform backpropagation if an optimizer is provided.

    Args:
        loss_func: Loss function.
        prediction: Model predictions.
        yb: Target labels.
        opt: Optimizer (default: None).

    Returns:
        Tuple[float, int]: Loss and number of samples in the batch.
    """
    loss = loss_func(prediction, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(yb)


def fit(config, model, loss_func, opt, train_dl, valid_dl, tracker_train, tracker_val):
    """
    Train the model.

    Args:
        config: Configuration object.
        model: PyTorch model.
        loss_func: Loss function.
        opt: Optimizer.
        train_dl: Training data loader.
        valid_dl: Validation data loader.
        tracker_train: Training metrics tracker.
        tracker_val: Validation metrics tracker.

    Returns:
        Tuple[nn.Module, list, list, Tracker, Tracker]: Trained model, training loss, validation loss,
            training metrics tracker, validation metrics tracker.
    """
    df = pd.DataFrame(columns=["Epoch", "Train_Loss", "Validation_Loss"])
    loss_arr_train = []
    loss_arr_val = []
    keep_training = True
    for epoch in range(config.epochs):
        tracker_train.increment()
        model.train()
        train_loss = 0
        num = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            prediction = model(xb)

            loss, len = loss_batch(loss_func, prediction, yb, opt)
            tracker_train.update(prediction, yb)

            train_loss += (loss * len)
            num += len

        train_loss = train_loss / num

        tracker_val.increment()
        model.eval()
        with torch.no_grad():
            val_loss = 0
            num = 0
            for xb, yb in valid_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                prediction = model(xb)

                loss, len = loss_batch(loss_func, prediction, yb)
                tracker_val.update(prediction, yb)

                val_loss += (loss * len)
                num += len

        val_loss = val_loss / num

        loss_arr_train.append(train_loss)
        loss_arr_val.append(val_loss)

        df2 = {'Epoch': epoch, 'Train_Loss': train_loss, 'Validation_Loss': val_loss}
        df = df.append(df2, ignore_index=True)

        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}| Validation loss: {val_loss:.5f}")

            filename = config.save_dir + "Loss_" + config.run_name + ".csv"
            df.to_csv(filename)

        keep_training = _early_stopping(config, model, loss_arr_val)

        if not keep_training and config.stop_early:
            print("Early Stopping")
            model.load_state_dict(torch.load(config.save_dir + config.run_name + '_checkpoint.pt'))
            break

    return model, loss_arr_train, loss_arr_val, tracker_train, tracker_val


def plot_losses(config, loss_arr_train, loss_arr_val):
    """
    Plot training and validation losses.

    Args:
        config: Configuration object.
        loss_arr_train: List of training losses.
        loss_arr_val: List of validation losses.
    """
    num_epochs_run = np.arange(1, len(loss_arr_train) + 1)

    fs = 20

    plt.plot(num_epochs_run, loss_arr_train, label=f"Training Loss", lw=2)
    plt.plot(num_epochs_run, loss_arr_val, label=f"Validation Loss", lw=2)
    plt.xlabel("Epoch", fontsize=fs)
    plt.title('batch size = ' + str(config.batch_size))
    plt.legend()
    plt.savefig(config.save_dir + config.run_name + '_Loss.png', dpi=600)
    plt.show()


def plot_other_metrics(config, total_train_metrics, total_val_metrics, met_list):
    """
    Plot additional metrics.

    Args:
        config: Configuration object.
        total_train_metrics: Training metrics tracker.
        total_val_metrics: Validation metrics tracker.
        met_list: List of additional metrics to plot.
    """
    num_epochs_run = np.arange(1, len(total_train_metrics.get(met_list[0])) + 1)

    fs = 20

    for met in met_list:
        train_plot = total_train_metrics.get(met).cpu().numpy()
        val_plot = total_val_metrics.get(met).cpu().numpy()
        plt.plot(num_epochs_run.squeeze(), train_plot, label=f"Training {met}", lw=2)
        plt.plot(num_epochs_run.squeeze(), val_plot, label=f"Validation {met}", lw=2)
        plt.xlabel("Epoch", fontsize=fs)
        plt.title('batch size = ' + str(config.batch_size))
        plt.legend()
        plt.savefig(config.save_dir + config.run_name + '_' + met + '.png', dpi=600)
        plt.show()
