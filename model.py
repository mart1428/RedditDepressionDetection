import torch
import torch.nn as nn
import torch.nn.functional as F

class DepressionDetector(nn.Module):
    def __init__(self, embed):
        super(DepressionDetector, self).__init__()
        self.name = 'DepressionDetector'
        self.emb = nn.Embedding.from_pretrained(embed.vectors, True)
        drop_prob = 0.3
        self.h_features = 256
        self.num_layers = 1

        self.lstm = nn.LSTM(300, self.h_features, self.num_layers, batch_first = True, dropout = drop_prob)
        self.gru = nn.GRU(300, 128, 2, batch_first = True, dropout = drop_prob)

        self.dropout = nn.Dropout(0.4)
        self.batchNorm = nn.LazyBatchNorm1d()
        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.LazyLinear(512)
        self.fc3 = nn.LazyLinear(2)

    def forward(self, x):
        x = self.emb(x)

        h0 = torch.zeros(self.num_layers, x.size(0), self.h_features, device = 'cuda' if torch.cuda.is_available() else 'cpu')
        c0 = torch.zeros(self.num_layers, x.size(0), self.h_features, device = 'cuda' if torch.cuda.is_available() else 'cpu')
        
        h0_1 = torch.zeros(2, x.size(0), 128, device = 'cuda' if torch.cuda.is_available() else 'cpu')

        x1, _ = self.lstm(x, (h0,c0))

        # x2, _ = self.gru(x, (h0_1))

        # x = torch.cat((x1, x2), dim = 1)

        x = self.dropout(F.relu(self.batchNorm(self.fc1(x1[:,-1,:]))))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x