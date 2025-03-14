import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import os
import json

from functions import *
from utils import *
from model import JointBERT

from transformers import BertTokenizer

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    hid_size = 768

    lr = 1e-5                                      # learning rate
    clip = 5                                       # Clip the gradients

    # convert the txt dataset into json dataset

    if not os.path.exists(os.path.join('dataset', 'laptop14_train.json')):
        convert_text_to_json(os.path.join('dataset', 'laptop14_train.txt'), os.path.join('dataset', 'laptop14_train.json'))
    if not os.path.exists(os.path.join('dataset', 'laptop14_test.json')):
        convert_text_to_json(os.path.join('dataset', 'laptop14_test.txt'), os.path.join('dataset', 'laptop14_test.json'))

    # load the dataset

    train_raw = load_data(os.path.join('dataset','laptop14_train.json'))
    test_raw = load_data(os.path.join('dataset','laptop14_test.json'))

    # select from the train dataset the 10% of the sample that will be the dev dataset

    portion = 0.10
    dev_raw = random.sample(train_raw, int(len(train_raw) * portion))
    dev_set = set(map(json.dumps, dev_raw))
    train_raw = [entry for entry in train_raw if json.dumps(entry) not in dev_set]

    slots = set(sum([line['slots'].split() for line in train_raw],[]))                      # since in develop set the slots are always 'O' and 'T', I compute the vocab only on the  
                                                                                            # training set (supposing there's at least one element for each class). 

    lang = Lang(slots)

    out_slot = len(lang.slot2id)                                                        # the model will output a vector of probabilities of length out_slot for each slot 

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")                      # download the tokenizer, I'll use BERT

    model = JointBERT(hid_size, out_slot).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.slot2id['pad'])             # loss function, ignoring 'pad' label




    train_dataset = Slots(train_raw, lang)
    dev_dataset = Slots(dev_raw, lang)
    test_dataset = Slots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = -1


    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        model, clip=clip)
        # if x % 5 == 0: # We check the performance every 5 epochs
        if True:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(dev_loader, criterion_slots, model, lang)

            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['f1']

            if f1 > best_f1:
                best_f1 = f1
                patience = 3
            else:
                patience -= 1
            if patience <= 0: 
                break 


    results_test, loss_dev = eval_loop(test_loader, criterion_slots, model, lang)    

    print('Slot F1: ', results_test['f1'])
    print('Slot precision: ', results_test['precision'])
    print('Slot recall: ', results_test['recall'])