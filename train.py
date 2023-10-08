import torch
import torch.nn as nn
import torch.optim as optim

import time

def get_model_name(name, batch_size, lr, epoch):
    path = 'model_{0}_bs{1}_lr{2}_epoch{3}'.format(name, batch_size, lr, epoch)
    return path

def train_model(model, train_loader, val_loader, batch_size = 128, lr = 0.001, epochs = 300):
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= 0.001)

    model = model.to(device)

    print(f'Training in {device}')

    for epoch in range(epochs):
        running_loss = 0
        running_error = 0
        total = 0
        corr = 0
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0

        model.train()

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            running_error += (predicted != labels).long().sum().item()
            corr += (predicted == labels).long().sum().item()
            true_pos += ((predicted == 1) & (labels == 1)).long().sum().item()
            false_pos += ((predicted == 1) & (labels == 0)).long().sum().item()
            false_neg += ((predicted == 0) & (labels == 1)).long().sum().item()
            true_neg += ((predicted == 0) & (labels == 0)).long().sum().item()
            total += labels.size(0)

        
        train_loss = running_loss/len(train_loader)
        train_error = running_error/len(train_loader.dataset)
        train_acc = corr/total

        with torch.no_grad():
            running_error = 0
            running_loss = 0
            total = 0
            corr = 0
            true_neg = 0
            true_pos = 0
            false_neg = 0
            false_pos = 0

            model.eval()

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                running_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)

                running_error += (predicted != labels).long().sum().item()
                corr += (predicted == labels).long().sum().item()
                true_pos += ((predicted == 1) & (labels == 1)).long().sum().item()
                false_pos += ((predicted == 1) & (labels == 0)).long().sum().item()
                false_neg += ((predicted == 0) & (labels == 1)).long().sum().item()
                true_neg += ((predicted == 0) & (labels == 0)).long().sum().item()
                total += labels.size(0)

            val_loss = running_loss/len(val_loader)
            val_error = running_error/len(val_loader.dataset)
            val_acc = corr/total

        print(f'Epoch {epoch+1}: Train | Loss: {train_loss:.3f}, Error: {train_error:.3f}, Acc: {train_acc:.3%} || Val | Loss: {val_loss:.3f}, Error: {val_error:.3f}, Acc: {val_acc:.3%}')
        end_time = time.time() - start_time
        print(f'Time after epoch {epoch+1}: {end_time:.2f}')
        file_path = get_model_name(model.name, batch_size, lr, epoch+1)
        torch.save(model.state_dict, file_path)