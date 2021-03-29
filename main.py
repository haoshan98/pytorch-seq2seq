import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import random
import os
import math
import time
from tqdm import tqdm

from data_loader import data_iterator
from model import Seq2seq

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Configs, Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
LEARNING_RATE = 1e-2
BATCH_SIZE = 128
CLIP = 1
EPOCHS = 30
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LOAD_MODEL = True
SAVE_MODEL = not LOAD_MODEL #True
os.makedirs('logs', exist_ok=True)

def train_fn(model, iterator, optimizer, criterion, clip, scaler, device, epoch):
    model.train()
    loop = tqdm(iterator, leave=True)
    loss_meter = 0
    start_time = time.time()
    for k, batch in enumerate(loop):
        src = batch.src
        trg = batch.trg # (seq, bs)
        src.to(device)
        trg.to(device) 

        # forward
        with torch.cuda.amp.autocast():
            pred = model(src, trg)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            trg = trg[1:].view(-1)
            output_dim = pred.shape[-1]
            pred = pred[1:].view(-1, output_dim)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(pred, trg) 
            loss_meter += loss.item()
        # backward
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(train_loss=loss_meter/(k+1))

    loss_meter /= len(iterator)

    return {'loss': loss_meter,
            'ppl': math.exp(loss_meter),
            'time': time.time() - start_time}

def valid_fn(model, iterator, criterion, device, epoch):
    model.eval()
    loop = tqdm(iterator, leave=True)
    loss_meter = 0
    pred_outputs = []
    start_time = time.time()
    for k, batch in enumerate(loop):
        src = batch.src
        trg = batch.trg # (seq, bs)
        src.to(device)
        trg.to(device) 

        # forward
        with torch.no_grad():
            pred = model(src, trg, 0) #turn off teacher forcing
            pred_outputs.append(pred)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            trg = trg[1:].view(-1)
            output_dim = pred.shape[-1]
            pred = pred[1:].view(-1, output_dim)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(pred, trg) 
            loss_meter += loss.item()

        # update progress bar
        if epoch < 0: # test set
            loop.set_description(f"Test [1/1]")
            loop.set_postfix(test_loss=loss_meter/(k+1))
        else:
            loop.set_description(f"Evaluate [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(valid_loss=loss_meter/(k+1))

    loss_meter /= len(iterator)

    return {'loss': loss_meter,
            'ppl': math.exp(loss_meter),
            'time': time.time() - start_time}, pred_outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    train_iterator, valid_iterator, test_iterator, SRC, TRG, test_data = \
        data_iterator(DEVICE, batch_size=BATCH_SIZE)
    en_vocab_size, de_vocab_size = len(SRC.vocab), len(TRG.vocab)

    model = Seq2seq(EMB_DIM, HID_DIM, en_vocab_size, de_vocab_size, N_LAYERS, ENC_DROPOUT, DEC_DROPOUT,
                    device=DEVICE).to(DEVICE)
    model.apply(init_weights)
    pad_index = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f'The model has {count_parameters(model):,} trainable parameters')

    history = [] 
    best_valid_loss = float('inf')
    def translate(pred_outputs, n_batch, n_sent):
            # get single batch, convert (seq len, bs, vocab size) -> (bs, seq len, vocab size)
            pred_output = pred_outputs[n_batch].permute(1, 0, 2)
            for i, sent in enumerate(pred_output): # iterate single batch
                if i < n_sent:
                    sent = sent.argmax(-1)
                    translation = ' '.join([TRG.vocab.itos[idx] for idx in sent[1:]])
                    print(f"{n_batch+1}-{i+1}: {translation} -> {sent.shape[0]}") # [seq_len]
    
    if LOAD_MODEL:
        checkpoint = torch.load("logs/seq2seq_checkpoint.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        test_res, test_pred_outputs = valid_fn(model, test_iterator, criterion, DEVICE, epoch=-1)
        for i in range(len(test_pred_outputs)):
            translate(test_pred_outputs, n_batch=i, n_sent=5)
        return

    for epoch in range(EPOCHS):
        
        train_res = train_fn(model, train_iterator, optimizer, criterion, CLIP, scaler, DEVICE, epoch)
        valid_res, pred_outputs = valid_fn(model, valid_iterator, criterion, DEVICE, epoch)
        scheduler.step(valid_res['loss'])
        
        res = {}
        for key in train_res: res[f'train_{key}'] = train_res[key]
        for key in valid_res: res[f'valid_{key}'] = valid_res[key]
        res['epoch'] = epoch
        history.append(res)
        histdf = pd.DataFrame(history)
        histdf.to_csv(f'logs/history.csv')

        if SAVE_MODEL and valid_res['loss'] < best_valid_loss:
            best_valid_loss = valid_res['loss']
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, "logs/seq2seq_checkpoint.pth.tar")
        
        for i in range(len(pred_outputs)-4):
            translate(pred_outputs, n_batch=i, n_sent=3)
    
    test_res, test_pred_outputs = valid_fn(model, test_iterator, criterion, DEVICE, epoch=0)    
    translate(test_pred_outputs, n_batch=2, n_sent=3)

 
if __name__ == "__main__":
    main()


