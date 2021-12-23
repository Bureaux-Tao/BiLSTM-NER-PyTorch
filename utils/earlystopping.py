import math

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience = 7, verbose = False, delta = 0, save_model = False, path = 'checkpoint.pth',
                 save_each_epoch = False, trace_func = print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_model = save_model
        self.save_each_epoch = save_each_epoch
    
    def __call__(self, val_loss, model, epoch = 0):
        
        score = -val_loss
        
        if self.save_each_epoch:
            self.save_checkpoint(val_loss, model, epoch)
        
        if self.best_score is None:
            self.best_score = score
            if not self.save_each_epoch:
                self.save_checkpoint(val_loss, model, epoch)
        elif math.isnan(float(val_loss)) or float(val_loss) == 0:
            self.trace_func(f'EarlyStopping due to monitor value is ' + str(val_loss))
            self.early_stop = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model and (not self.save_each_epoch):
                self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.save_each_epoch:
            if self.path.count('.') > 1:
                raise Exception("saved path contains more than one '.'")
            name = self.path.split('.')[-2]
            extension = self.path.split('.')[-1]
            saved_path = name + '_ep' + str(epoch) + '.' + extension
            if self.verbose:
                self.trace_func(
                    f'Saving model in path {self.path} on epoch {str(epoch)}.')
            torch.save(model.state_dict(), saved_path)
            self.val_loss_min = val_loss
        else:
            if self.verbose:
                self.trace_func(
                    f'Monitor value decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
