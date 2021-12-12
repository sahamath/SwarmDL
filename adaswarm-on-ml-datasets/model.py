import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self, n_features, n_neurons, n_out):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(in_features=n_features, out_features=n_neurons)
        self.out_layer = torch.nn.Linear(in_features=n_neurons, out_features=n_out)

    def forward(self, X):
        out = F.relu(self.hidden(X))
        out = F.sigmoid(self.out_layer(out))
        return out

class TwoLayerModel(torch.nn.Module):
    def __init__(self, n_features, n_neurons, n_out):
        super(TwoLayerModel, self).__init__()
        self.hidden_1 = torch.nn.Linear(in_features=n_features, out_features=n_neurons)
        self.hidden_2 = torch.nn.Linear(in_features=n_neurons, out_features=n_neurons)
        self.out_layer = torch.nn.Linear(in_features=n_neurons, out_features=n_out)

    def forward(self, X):
        out = F.relu(self.hidden_1(X))
        out = F.relu(self.hidden_2(out))
        out = F.sigmoid(self.out_layer(out))
        return out

class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output