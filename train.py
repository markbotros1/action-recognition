import os
import glob
import yaml
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import evaluate
from transformers import VivitForVideoClassification, get_scheduler

from dataset import NFLDataset

def train(epochs, tr_loader, val_loader, model, optimizer, device, 
          train_iters, eval_iters, eval_every, grad_accum_iter):
    
    dev = torch.device(device)
    model.to(dev)

    n_steps = epochs * len(tr_loader) * grad_accum_iter
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer,
        num_warmup_steps=0, 
        num_training_steps=n_steps
    )
    
    best_acc = 0
    for epoch in range(epochs):
        for i in tqdm(range(train_iters)):
            model.train()
            _, X, y = next(iter(tr_loader))
            outputs = model(pixel_values=X.to(device), labels=y.to(device))
            loss = outputs.loss
            loss = loss / grad_accum_iter 
            loss.backward()
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
            if ((i + 1) % grad_accum_iter == 0) or ((i + 1) == len(tr_loader)):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (i + 1) % eval_every == 0:
                print('Validating...')
                acc = eval(val_loader, model, device, eval_iters)
                print(f'    Accuracy: {acc}')
                if acc > best_acc:
                    model.save_pretrained(os.path.join('.', 'models'))
                    best_acc = acc

            torch.cuda.empty_cache()
    return model


def eval(val_loader, model, device, eval_iters):
    metric = evaluate.load('accuracy')
    model.eval()
    for i in range(eval_iters):
        _, X, y = next(iter(val_loader))
        with torch.no_grad():
            outputs = model(pixel_values=X.to(device), labels=y.to(device))

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=y)
    return metric.compute()['accuracy']


def load_config(cnf_path, cnf_name):
    with open(os.path.join(cnf_path, cnf_name)) as file:
        config = yaml.safe_load(file)
    return config


def load_data(cnf):
    helmets = pd.read_csv(os.path.join(cnf['data_dir'], cnf['helmets']))
    metadata = pd.read_csv(os.path.join(cnf['data_dir'], cnf['metadata']),
                           parse_dates=cnf['parse_cols_m'])
    labels = pd.read_csv(os.path.join(cnf['data_dir'], cnf['labels']), 
                         parse_dates=cnf['parse_cols_l'])
    if 2 not in labels['contact'].unique():
        labels.loc[
            (labels['contact'] == 1) & (labels['nfl_player_id_2'] == 'G'), 
            'contact'
        ] = 2
        labels.to_csv('data/train_labels.csv', index=False)
    return helmets, labels, metadata



if __name__ == '__main__':
    print('Loading configs & data')
    cnf = load_config('', 'config.yaml')
    helmets, labels, metadata = load_data(cnf)


    print('Creating dataset & dataloader')
    rus = RandomUnderSampler()
    x, y = rus.fit_resample(labels, labels['contact'])
    xTr, xTe, yTr, yTe = train_test_split(x, y, test_size=.2)

    tr_dataset = NFLDataset(xTr, helmets, metadata, cnf['video_dir'], cnf['img_size'],  
                            cnf['start'], cnf['stop'], cnf['interval'])
    val_dataset = NFLDataset(xTe, helmets, metadata, cnf['video_dir'],  cnf['img_size'], 
                            cnf['start'], cnf['stop'], cnf['interval'])

    tr_loader = DataLoader(tr_dataset, batch_size=cnf['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cnf['batch_size'])


    print('Loading pretrained model')
    pretrained_models = glob.glob('./models/*.bin')
    model = None
    if len(pretrained_models) == 0:
        model = VivitForVideoClassification.from_pretrained(
            'google/vivit-b-16x2-kinetics400',
            num_frames=cnf['stop'],
            num_labels=cnf['n_labels'],
            ignore_mismatched_sizes=True
        )
    else:
        model = VivitForVideoClassification.from_pretrained(
            './models/.',
            num_frames=cnf['stop'],
            num_labels=cnf['n_labels'],
            ignore_mismatched_sizes=True
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    total_params = sum(p.numel() for p in model.parameters())
    print('# Trainable Parameters:', total_params)

    
    print('Finetuning')
    model = train(cnf['epochs'],
            tr_loader=tr_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            device=cnf['device'],
            train_iters=cnf['train_iters'],
            eval_iters=cnf['eval_iters'],
            eval_every=cnf['eval_every'],
            grad_accum_iter=cnf['grad_accum_iter'])
    
