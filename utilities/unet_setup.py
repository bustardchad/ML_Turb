# Helper functions for U-net models
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from unet import UNet

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up loss function and prediction function with metrics to track
from torchmetrics import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

def ssim_loss(x, y, alpha):
    return alpha*(1. - ssim(x, y))

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
        
def loss_batch(config, loss_func, prediction, yb, opt=None, struc_alpha = 0.2):
    loss = loss_func(prediction, yb)
    if config.use_ssim:
      struc_loss = ssim_loss(prediction, yb, struc_alpha)
      loss += struc_loss
    else:
      struc_loss = 0
 
    #print(f"Metrics on batch: {batch_metrics}")

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), struc_loss.item(), len(yb)


def fit(config, model, loss_func, opt, train_dl, valid_dl, alpha = 0.2):
    # Function that fits (trains) the network
    #
    # Returns: model, loss arrays for training and validation, 
    #           structural similarity measures for training and validation
    #######################################################################

    # arrays to store losses
    loss_arr_train = []
    loss_arr_val = []
    struc_arr_train = []
    struc_arr_val = []
    keep_training = True
    for epoch in range(config.epochs):
        model.train()
        train_loss= 0
        train_struc = 0
        num = 0
        for xb, yb, lbl in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            prediction = model(xb)
            
            loss, struc_loss, len = loss_batch(config, loss_func, prediction, 
                                               yb, opt, alpha)
            if config.use_ssim:
              train_struc += -(struc_loss/alpha - 1.0)*len
            
            train_loss += (loss*len)
            num += len

        train_loss = train_loss/num
        if config.use_ssim:
          train_struc = train_struc/num # average SSIM value (1.0 is best)

        #print(f"current training stats={tracker_train.compute()}") 

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
              val_struc += -(struc_loss/alpha - 1.0)*len
            
            val_loss += (loss*len)
            num += len

        if config.use_ssim:
          val_struc = val_struc/num
        
        val_loss = val_loss/num    

        #print(f"current validation stats={tracker_val.compute()}") 
        
        loss_arr_train.append(train_loss)
        loss_arr_val.append(val_loss)

        if config.use_ssim:
          struc_arr_train.append(train_struc)
          struc_arr_val.append(val_struc)


        # Print out what's happening every epoch
        if epoch % 1 == 0:
          print(f"Epoch: {epoch} | Train Loss: {train_loss:.5f}| Validation loss: {val_loss:.5f}")

          # optionally plot results on validation data every epoch
          model_for_eval = model
          plot_results(config, model_for_eval.cpu(), valid_dl, epoch)
          model = model.to(device)

        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        keep_training = _early_stopping(config, model, loss_arr_val)
        
        if ((keep_training==False) and (config.stop_early==True)):
          print("Early Stopping")
          # load the last checkpoint with the best model
          model.load_state_dict(torch.load(config.save_dir+config.run_name+'_checkpoint.pt'))
          break

    return model, loss_arr_train, loss_arr_val, struc_arr_train, struc_arr_val


def create_model(config, n_blocks, start_filts, lr, weight_decay):
    # Creates a Unet with input parameters for # of blocks, starting # of filters,
    # learning rate, and weight decay rate
    model = UNet(in_channels=1,
             out_channels=1,
             #n_blocks=3,
             n_blocks=n_blocks,
             start_filts=start_filts,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

    loss_fn = nn.MSELoss() #default reduction='mean'

    opt = torch.optim.AdamW(model.parameters(),
                                    lr = lr, 
                                    weight_decay = weight_decay)

    return model, loss_fn, opt


def plot_results(config, model_eval, data_loader, epoch):
    # Plots image batches after U-net mapping
    #
    # Rows = input images, target images, and network generated images

    #model_eval = model_cuda.to('cpu')
    model_eval.eval()

    # access a batch of labelled images
    dataiter = iter(data_loader)
    x_arr, y_arr, labels = next(dataiter)

    n_examples = 6
    fig, axs = plt.subplots(3,n_examples,figsize=(int(8*n_examples/3),8))
    with plt.style.context('fast'):
        for i in range(0,n_examples):
            x = x_arr[i]
            y = y_arr[i]
            # requires a 4D tensor, so need to reshape this 3D one
            image = x.reshape(1, 1, x.shape[1], x.shape[2])
            target = y.reshape(1, 1, y.shape[1], y.shape[2])
            
            target = target.reshape(target.shape[2],target.shape[3])
            target = target.detach().numpy()


            # we need to find the gradient with respect to the input image, so we need to call requires_grad_  on it
            image.requires_grad_()
            
            # run the model on the image
            outputs = model_eval(image)

            #print(ssim(y.reshape(1,1,128,128), outputs))
            #print(loss_fn(y.reshape(1,1,128,128), outputs))

            input_image = image.reshape(image.shape[2],image.shape[3])
            input_image = input_image.detach().numpy()

            target_image = outputs.reshape(image.shape[2],image.shape[3])
            target_image = target_image.detach().numpy()
            
            axs[0,i].imshow(input_image,cmap='gray')
            axs[1,i].imshow(target,cmap='gray')
            axs[2,i].imshow(target_image,cmap='gray')
            
        
            cols = ['Example {}'.format(col) for col in range(1, n_examples+1)]
            rows = ['Input', 'Real Target', 'Modeled Target']

            for ax, col in zip(axs[0], cols):
                ax.set_title(col)

            for ax, row in zip(axs[:,0], rows):
                ax.set_ylabel(row, rotation=90, size='large')

            fig.tight_layout()
            plt.show()
            plt.savefig(config.save_dir+config.run_name+'_image_epoch'+str(epoch)+'.png',dpi=600)
            plt.close()