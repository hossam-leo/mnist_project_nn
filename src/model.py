import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10, activation='relu', use_dropout=False):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.2)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = F.relu

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc2(x)
        return x
