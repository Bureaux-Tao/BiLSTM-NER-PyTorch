import os
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import path
from metric import NER_Metric
from utils.earlystopping import EarlyStopping

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(model, optimizer, train_dl, epoch, device):
    steps = 0
    total_loss = 0.
    batch_size = train_dl.batch_size
    
    total_steps = int(len(train_dl.dataset) / batch_size)
    
    loop = tqdm(enumerate(train_dl), total = len(train_dl))
    for step, (sequence, tags) in loop:
        optimizer.zero_grad()
        
        sequence_cuda = sequence.to(device)
        tags_cuda = tags.to(device)
        mask_cuda = sequence_cuda > 0
        
        loss = model(sequence_cuda, tags_cuda, mask_cuda)
        
        loss.backward()
        
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        steps += 1
        
        train_loss = total_loss / (steps * batch_size)
        loop.set_description(f'epoch [{epoch}]')
        loop.set_postfix(loss = train_loss)
        
        if steps % 400 == 0:
            print('epoch: {} - progress: {:.4f}'.format(epoch, steps / total_steps))
    
    train_loss = total_loss / (steps * batch_size)
    
    return train_loss


def train(model, optimizer, train_dl, test_dl, device = None, epochs = 500, reduce_lr_epochs = 3,
          early_stop_epochs = 10, weights_save_path = path.saved_model_name):
    history = {
        'epoch': [],
        'acc': [],
        'loss': [],
        'test_acc': [],
    }
    
    reduceLR = ReduceLROnPlateau(optimizer, factor = 0.5, patience = reduce_lr_epochs, min_lr = 1e-9, verbose = True)
    early_stopping = EarlyStopping(patience = early_stop_epochs, verbose = False, save_model = True,
                                   path = path.weights_path + weights_save_path)
    
    print('training on ', device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        train_loss = train_one_epoch(model, optimizer, train_dl, epoch, device)
        
        with torch.no_grad():
            val_metric = evaluate(model, test_dl, device = device)
            train_metric = evaluate(model, train_dl, device = device)
            
            train_acc = train_metric.global_precision
            test_acc = val_metric.global_precision
        
        history['epoch'].append(epoch)
        history['acc'].append(train_acc)
        history['loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        
        logger.info(
            "epoch {} - loss: {:.4f} acc: {:.4f} - test_acc: {:.4f}\n".format(epoch, train_loss, train_acc, test_acc))
        
        early_stopping(train_loss, model)
        
        if early_stopping.early_stop:
            print("\nEarly stopping after epoch " + str(epoch))
            break
        
        reduceLR.step(train_loss)
    
    return history


def predict(model, dl, device = None):
    tags_pred_list = []
    with torch.no_grad():
        for sequence, tags in dl:
            sequence_cuda = sequence.to(device)
            mask_cuda = sequence_cuda > 0
            
            tags_pred = model.predict(sequence_cuda, mask_cuda)
            tags_pred.extend(tags_pred_list)
    
    return tags_pred_list


def evaluate(model, dl, device = None, toast = 'predicting...', verbose = False):
    model.eval()
    
    metric = NER_Metric()
    
    with torch.no_grad():
        
        if verbose:
            loop = tqdm(enumerate(dl), total = len(dl))
            for step, (sequence, tags) in loop:
                loop.set_description(toast)
                sequence_cuda = sequence.to(device)
                mask_cuda = sequence_cuda > 0
                
                tags_pred = model.predict(sequence_cuda, mask_cuda)
                
                for i in range(len(tags)):
                    metric.update(tags[i].numpy(), tags_pred[i])
        
        else:
            for sequence, tags in dl:
                sequence_cuda = sequence.to(device)
                mask_cuda = sequence_cuda > 0
                
                tags_pred = model.predict(sequence_cuda, mask_cuda)
                
                for i in range(len(tags)):
                    metric.update(tags[i].numpy(), tags_pred[i])
    
    return metric


def save_model(model, epoch, acc):
    model_file_name = 'model_{}_epoch_{}_acc_{:.2f}'.format(model.name, epoch, acc)
    
    model_dir = './model/'
    save_path = os.path.join(model_dir, '{}.tar'.format(model_file_name))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'model': model.state_dict(),
    }, save_path)


def load_model(model, model_file_name):
    model_dir = './model/'
    model_file = os.path.join(model_dir, model_file_name)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model'])
