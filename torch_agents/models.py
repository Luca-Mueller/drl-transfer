import torch
from torch import nn
import torch.nn.functional as F


# DQN Model
class DQN(nn.Module):
    def __init__(self, inputs: int, outputs: int, fc1_nodes: int = 32, fc2_nodes: int = 32):
        super(DQN, self).__init__()
        self.shape = (inputs, outputs)
        self.fc1_nodes = fc1_nodes
        self.fc2_nodes = fc2_nodes

        self.fc1 = nn.Linear(inputs, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.fc3 = nn.Linear(fc2_nodes, outputs)

    def forward(self, x):
        x = x.squeeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x, device: str):
        y = self.__call__(x).max(0)[1].view(1, 1)
        return torch.tensor([[y]], dtype=torch.long, device=device)


# V Model
class VModel(nn.Module):
    def __init__(self, inputs: int, fc1_nodes: int, fc2_nodes: int):
        super(VModel, self).__init__()

        self.fc1 = nn.Linear(inputs, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.fc3 = nn.Linear(fc2_nodes, 1)

    def forward(self, x):
        x = x.squeeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# V2 Model
class V2Model(nn.Module):
    def __init__(self, dqn: DQN):
        super(V2Model, self).__init__()

        self.fc1 = dqn.fc1
        self.fc2 = dqn.fc2
        self.fc3 = nn.Linear(dqn.fc2_nodes, 1)

    def forward(self, x):
        x = x.squeeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def freeze(self):
        self.fc1.weight.requires_grad = False
        self.fc2.weight.requires_grad = False

    def unfreeze(self):
        self.fc1.weight.requires_grad = True
        self.fc2.weight.requires_grad = True
