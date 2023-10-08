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

def test_model(model, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        running_error = 0
        running_loss = 0

        correct = 0
        total = 0

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            running_loss += criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            running_error += (predicted != labels).long().sum().item()
            correct += (predicted == labels).long().sum().item()
            total += labels.size(0)

            true_pos += ((predicted == 1) & (labels == 1)).long().sum().item()
            false_pos += ((predicted == 1) & (labels == 0)).long().sum().item()
            false_neg += ((predicted == 0) & (labels == 1)).long().sum().item()
            true_neg += ((predicted == 0) & (labels == 0)).long().sum().item()

        avg_test_loss = running_loss/len(test_loader)
        avg_test_error = running_error/len(test_loader.dataset)
        acc = correct/total

        recall = true_pos/(true_pos+false_neg)
        precision = true_pos/(true_pos+false_pos)

    print(f'Test| Loss: {avg_test_loss:4f}, Error: {avg_test_error:4f}, Acc: {acc:4%}%, Precision: {precision:.4%}, Recall: {recall:.4%}')


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    batch_size = 128

    embed = GloVe(name = '840B', dim = 300, cache = 'C:\\Users\\chris\\Documents\\Online Courses & Personal Projects\\Portfolio\\RNN-LSTM\\.vector_cache')
    train_loader, val_loader, test_loader = get_data_loader(embed, batch_size=batch_size)
    
    model = DepressionDetector(embed)

    lr = 0.001
    epochs = 30
    train_model(model, train_loader, val_loader, batch_size, lr, epochs)
    test_model(model, test_loader)