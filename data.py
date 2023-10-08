import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchtext.vocab as vocab

import pandas as pd
import sys, string

def get_data_loader(embed, batch_size = 128):
    data = pd.read_csv('depression_dataset_reddit_cleaned.csv')
    d = {1 : data[data['is_depression'] == 1].index , 0 : data[data['is_depression'] == 0].index}


    train_indices = []
    val_indices = []
    test_indices = []

    for k, v in d.items():
        split1 = int(0.85 * len(v))
        split2 = int(0.95 * len(v))

        train_indices += v[:split1].to_list()
        val_indices += v[split1:split2].to_list()
        test_indices += v[split2:].to_list()

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    
    # glove = vocab.GloVe(name = '840B', dim = 300, cache = 'C:\\Users\\chris\\Documents\\Online Courses & Personal Projects\\Portfolio\\RNN-LSTM\\.vector_cache')
    
    cleaned_data = []

    for i in data.values.tolist():
        inputs, labels = i

        max_length = 150
        tokenized_sequence = []

        for i in range(max_length):
            if i >= len(inputs):
                break
            t = inputs[i].translate(str.maketrans('', '', string.punctuation))
            if t and t in embed.stoi:
                tokenized_sequence.append(embed.stoi[t])
            else:
                tokenized_sequence.append(embed.stoi['unk'])

        if len(tokenized_sequence) <= max_length:
            tokenized_sequence += [embed.stoi['pad']] * (max_length-len(tokenized_sequence))

        cleaned_data.append([torch.tensor(tokenized_sequence), labels])
    
    train_loader = DataLoader(cleaned_data, batch_size, sampler = train_sampler)
    val_loader = DataLoader(cleaned_data, batch_size, sampler = val_sampler)
    test_loader = DataLoader(cleaned_data, batch_size, sampler = test_sampler)

    return train_loader, val_loader, test_loader