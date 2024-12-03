from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x