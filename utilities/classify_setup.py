# Helper functions for setting up the classification network
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


# if val_loss decreases, write a checkpoint of model and return true
# if val_loss has increased for -patience- number of steps, return false
def _early_stopping(config, model, val_loss):
  if (len(val_loss) > 1):
    if (val_loss[-1] < val_loss[-2]):
      print("Saving model checkpoint")
      torch.save(model.state_dict(), config.save_dir + config.run_name+'_checkpoint.pt')
      return True
  if (len(val_loss) > config.patience):
    if ((np.diff(val_loss[-config.patience:]) <= 0).all()):
      print("Early Stopping")
      return False
        

def loss_batch(loss_func, prediction, yb, opt=None):
    loss = loss_func(prediction, yb)
 
    #print(f"Metrics on batch: {batch_metrics}")

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(yb)

def fit(config, model, loss_func, opt, train_dl, valid_dl, tracker_train, tracker_val):
    # Function that fits (trains) the network
    #
    # Returns: model, loss arrays for training and validation
    #######################################################################
    #

    # text file where I'll write out losses
    # create a DataFrame
    df = pd.DataFrame(columns = ["Epoch", "Train_Loss", "Validation_Loss"])

    # arrays to store losses
    loss_arr_train = []
    loss_arr_val = []
    keep_training = True
    for epoch in range(config.epochs):
        tracker_train.increment()
        model.train()
        train_loss= 0
        num = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            prediction = model(xb)

            loss, len = loss_batch(loss_func, prediction, yb, opt)
            tracker_train.update(prediction, yb)

            train_loss += (loss*len)
            num += len

        train_loss = train_loss/num

        #print(f"current training stats={tracker_train.compute()}") 

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

            val_loss += (loss*len)
            num += len

        val_loss = val_loss/num    

        #print(f"current validation stats={tracker_val.compute()}") 
        
        loss_arr_train.append(train_loss)
        loss_arr_val.append(val_loss)

        df2 = {'Epoch': epoch, 'Train_Loss': train_loss, 'Validation_Loss': val_loss}

        # update dataframe holding losses
        df = df.append(df2, ignore_index = True)


        # Print out what's happening every epoch
        if epoch % 1 == 0:
          print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}| Validation loss: {val_loss:.5f}")

          # add loss (and SSIM info) to text file
          filename = config.save_dir+"Loss_"+config.run_name+".csv"
          df.to_csv(filename)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        keep_training = _early_stopping(config, model, loss_arr_val)
        
        if ((keep_training==False) and (config.stop_early==True)):
          print("Early Stopping")
          # load the last checkpoint with the best model
          model.load_state_dict(torch.load(config.save_dir + config.run_name+'_checkpoint.pt'))
          break

    return model, loss_arr_train, loss_arr_val, tracker_train, tracker_val

def plot_losses(config, loss_arr_train, loss_arr_val):
  # assumes training and validation loss are each quantified at every epoch
  num_epochs_run = np.arange(1,len(loss_arr_train)+1)

  # plotting parameters
  fs = 20

  plt.plot(num_epochs_run, loss_arr_train, label = f"Training Loss", lw=2)
  plt.plot(num_epochs_run, loss_arr_val, label = f"Validation Loss", lw=2)
  plt.xlabel("Epoch", fontsize=fs)
  plt.title('batch size = ' + str(config.batch_size))
  plt.legend()
  plt.savefig(config.save_dir+config.run_name+'_Loss.png',dpi=600)
  plt.show()
  #plt.close()

def plot_other_metrics(config, total_train_metrics, total_val_metrics, met_list):
  # assumes training and validation loss are each quantified at every epoch
  num_epochs_run = np.arange(1,len(total_train_metrics.get(met_list[0]))+1)

  # plotting parameters
  fs = 20

  
  for met in met_list:
    train_plot = total_train_metrics.get(met).cpu().numpy()
    val_plot = total_val_metrics.get(met).cpu().numpy()
    plt.plot(num_epochs_run.squeeze(), train_plot, label = f"Training {met}", lw=2)
    plt.plot(num_epochs_run.squeeze(), val_plot, label = f"Validation {met}", lw=2)
    plt.xlabel("Epoch", fontsize=fs)
    plt.title('batch size = ' + str(config.batch_size))
    plt.legend()
    plt.savefig(config.save_dir+config.run_name+'_'+met+'.png',dpi=600)
    plt.show()
    #plt.close()