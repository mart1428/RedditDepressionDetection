import torch
import torch.nn as nn

import torchtext
from torchtext import transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.vocab import GloVe

import numpy as np

from data import get_data_loader
from model import DepressionDetector
from train import train_model

if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    batch_size = 128

    embed = GloVe(name = '840B', dim = 300, cache = 'C:\\Users\\chris\\Documents\\Online Courses & Personal Projects\\Portfolio\\RNN-LSTM\\.vector_cache')
    train_loader, val_loader, test_loader = get_data_loader(embed, batch_size=batch_size)
    
    model = DepressionDetector(embed)

    lr = 0.001
    epochs = 300
    train_model(model, train_loader, val_loader, batch_size, lr, epochs)