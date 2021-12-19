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


def train_one_epoch(model, optimizer, train_dl, epoch, device, clip_grad):
    model.train()
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
        
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
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


def validate_one_epoch(model, optimizer, val_dl, device):
    model.eval()
    steps = 0
    total_loss = 0.
    batch_size = val_dl.batch_size
    
    with torch.no_grad():
        for sequence, tags in val_dl:
            sequence_cuda = sequence.to(device)
            tags_cuda = tags.to(device)
            mask_cuda = sequence_cuda > 0
            
            loss = model(sequence_cuda, tags_cuda, mask_cuda)
            total_loss += loss.item()
            
            steps += 1
    val_loss = total_loss / (steps * batch_size)
    return val_loss


def train(model, optimizer, train_dl, val_dl, device = None, epochs = 500, reduce_lr_epochs = 3,
          early_stop_epochs = 10, weights_save_path = "checkpoint.pth", save_each_epoch = False,
          clip_grad = 0.0):
    history = {
        'epoch': [],
        'acc': [],
        'loss': [],
        'val_acc': [],
        'val_loss': []
    }
    
    reduceLR = ReduceLROnPlateau(optimizer, factor = 0.5, patience = reduce_lr_epochs, min_lr = 1e-9, verbose = True)
    early_stopping = EarlyStopping(patience = early_stop_epochs, verbose = False, save_model = True,
                                   path = path.weights_path + weights_save_path, save_each_epoch = save_each_epoch)
    
    print('training on ', device)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_dl, epoch, device, clip_grad)
        val_loss = validate_one_epoch(model, optimizer, val_dl, device)
        
        with torch.no_grad():
            val_metric = evaluate(model, val_dl, device = device)
            train_metric = evaluate(model, train_dl, device = device)
            
            train_acc = train_metric.global_precision
            val_acc = val_metric.global_precision
        
        history['epoch'].append(epoch)
        history['acc'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info("epoch {} - loss: {:.4f} acc: {:.4f} val_loss: {:.4f} val_acc: {:.4f}\n".
                    format(epoch, train_loss, train_acc, val_loss, val_acc))
        
        early_stopping(val_loss, model, epoch = epoch)
        
        if early_stopping.early_stop:
            print("\nEarly stopping after epoch " + str(epoch))
            break
        
        reduceLR.step(val_loss)
    
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
