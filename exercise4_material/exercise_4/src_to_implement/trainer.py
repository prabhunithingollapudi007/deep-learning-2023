import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import platform
t.cuda.empty_cache()
import os, time
from matplotlib import pyplot as plt

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 scheduler = None,
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._scheduler = scheduler

        self._early_stopping_patience = early_stopping_patience
        
        # self._train_dl = [self.img_enhancement(img) for img in self._train_dl]
        # self._val_test_dl = [self.img_enhancement(img) for img in self._val_test_dl]
        
        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
        
        self.model_folder = os.getcwd()+'/saved_models/'
        self.timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        
    def save_checkpoint(self, epoch):
        if not os.path.exists(self.model_folder + self.timestamp):
            os.makedirs(self.model_folder + self.timestamp)
            self.model_folder_timestamped = self.model_folder + self.timestamp
        t.save({'state_dict': self._model.state_dict()}, self.model_folder_timestamped +'/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load(self.model_folder +'checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
    
    
    def calculate_metrics(self, total_loss, y_preds, y_grounds):
        '''
        Function to calculate evaluation metrics like loss and f1
        '''
        avg_loss = total_loss / len(self._val_test_dl)
        f1_crack = f1_score(y_true=y_grounds[:, 0, 0], y_pred=y_preds[:, 0, 0], average='binary')
        f1_inactive = f1_score(y_true=y_grounds[:, 0, 1], y_pred=y_preds[:, 0, 1], average='binary')
        f1_mean = (f1_crack + f1_inactive) / 2
        return avg_loss, f1_crack, f1_inactive, f1_mean
    
    def train_step(self, x, y):        
        self._optim.zero_grad() # Reset gradients
        x = self._model.forward(x) # Propagate
        calc_loss = self._crit(x,y) # The calculated loss
        calc_loss.backward() # Compute gradients
        self._optim.step() # does the update
        return calc_loss
        

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        preds = self._model.forward(x)
        loss = self._crit(preds, y.float())
        return loss, preds
    

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        losses = []
        self._model.train()

        for image, labels in self._train_dl:
            if self._cuda:
                image, labels = image.cuda(), labels.cuda()
            image = image.to(t.float32)
            labels = labels.to(t.float32)
            train_loss = self.train_step(image, labels)
            losses.extend([train_loss])
        
        for s in self._scheduler:
            s.step()
            
        avg_loss = sum(losses)/len(losses)
        return avg_loss
    
    @t.no_grad()       
    def val_test(self):
        self._val_test_dl.mode = "val"
        self._model.eval()
        
        y_preds = []
        y_grounds = []

        total_loss = 0

        # Iterate through the validation set
        for x, y in self._val_test_dl:
            x, y = x.cuda(), y.cuda()
            loss, y_pred = self.val_test_step(x, y)
            total_loss += loss.item()

            # Save the predictions and the labels for each batch
            y_preds.append(np.around(y_pred.cpu().numpy()))
            y_grounds.append(y.cpu().numpy())
        

        # Calculate relevant metrics and return them
        return self.calculate_metrics(total_loss, np.array(y_preds), np.array(y_grounds))
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        val_f1_mean = []
        val_f1_crack = []
        val_f1_inactive = []
        epoch_counter = 0
        
        es_check = 0 # Early stopping check: will count the times the validation loss was not decreasing
        previous_loss = self.val_test()[0]

        while epoch_counter<=epochs:
            # stop by epoch number # TO discuss
            # if epoch_counter > epochs:
            #     break
            # with t.no_grad():
            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            val_loss, f1_crack, f1_inactive, f1_mean = self.val_test()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_mean.append(f1_mean)
            val_f1_crack.append(f1_crack)
            val_f1_inactive.append(f1_inactive)
            
            if f1_mean > 0.6:
                self.save_checkpoint(epoch_counter) # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)

            if val_loss > previous_loss:
                es_check += 1 # Count how many times the validation loss does NOT increase
            
            if es_check >= self._early_stopping_patience:
                print('Patience exceeded. Early stopping.')
                break

            epoch_counter += 1
            print("Epoch: {}, \nTrain_loss: {:.2f}, Val_loss: {:.2f}, Val_f1_mean: {:.2f}, Val_f1_crack: {:.2f}, Val_f1_inactive: {:.2f}".format(
                epoch_counter,
                train_loss,
                val_loss,
                f1_mean,
                f1_crack,
                f1_inactive
                
            ))
        
        # Bringing the list to cpu from cuda (for plotting purposes)
        train_loss_cpu = t.tensor(train_losses, device = 'cpu')
        val_loss_cpu = t.tensor(val_losses, device = 'cpu')

        # plot the results
        plt.plot(np.arange(len(train_loss_cpu)), train_loss_cpu, label='train loss')
        plt.plot(np.arange(len(val_loss_cpu)), val_loss_cpu, label='val loss')
        plt.legend()
        plt.savefig(self.model_folder+'/loss')
        
        return train_losses, val_losses
                    
        
        
        
