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
        x = x.squeeze(0).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x, device: str):
        y = self.__call__(x).max(0)[1].view(1, 1)
        return torch.tensor([[y]], dtype=torch.long, device=device)

    def reset_head(self, outputs: int = None):
        if not outputs:
            self.fc3.reset_parameters()
        else:
            self.fc3 = nn.Linear(self.fc2_nodes, outputs)


# V Model
class VModel(nn.Module):
    def __init__(self, inputs: int, fc1_nodes: int, fc2_nodes: int):
        super(VModel, self).__init__()

        self.fc1 = nn.Linear(inputs, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.fc3 = nn.Linear(fc2_nodes, 1)

    def forward(self, x):
        x = x.squeeze(0).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_head(self, outputs: int = None):
        if not outputs:
            self.fc3.reset_parameters()
        else:
            self.fc3 = nn.Linear(self.fc2_nodes, outputs)


# V2 Model
class V2Model(nn.Module):
    def __init__(self, dqn: DQN):
        super(V2Model, self).__init__()

        self.fc1 = dqn.fc1
        self.fc2 = dqn.fc2
        self.fc3 = nn.Linear(dqn.fc2_nodes, 1)

    def forward(self, x):
        x = x.squeeze(0).float()
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

    def reset_head(self, outputs: int = None):
        if not outputs:
            self.fc3.reset_parameters()
        else:
            self.fc3 = nn.Linear(self.fc2_nodes, outputs)


# Conv DQN for ALE
class ConvDQN(nn.Module):
    def __init__(self, h: int, w: int, outputs: int, window_size: int = 4):
        super(ConvDQN, self).__init__()
        self.shape = ((window_size, h, w), outputs)
        self.conv1 = nn.Conv2d(window_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)

    def predict(self, x, device: str):
        y = self.__call__(x).max(1)[1].view(1, 1)
        return torch.tensor([[y]], dtype=torch.long, device=device)

    def reset_head(self, outputs: int = None):
        if not outputs:
            self.head.reset_parameters()
        else:
            self.head = nn.Linear(512, outputs)


# Conv V Model
class ConvVModel(nn.Module):
    def __init__(self, h: int, w: int, window_size: int = 4):
        super(ConvVModel, self).__init__()
        self.conv1 = nn.Conv2d(window_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)

    def reset_head(self, outputs: int = None):
        if not outputs:
            self.head.reset_parameters()
        else:
            self.head = nn.Linear(512, outputs)


# Conv V2 Model
class ConvV2Model(nn.Module):
    def __init__(self, conv_dqn: ConvDQN):
        super(ConvV2Model, self).__init__()
        self.conv1 = conv_dqn.conv1
        self.bn1 = conv_dqn.bn1
        self.conv2 = conv_dqn.conv2
        self.bn2 = conv_dqn.bn2
        self.conv3 = conv_dqn.conv3
        self.bn3 = conv_dqn.bn3
        self.fc1 = conv_dqn.fc1
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)

    def freeze(self):
        self.conv1.weight.requires_grad = False
        self.bn1.weight.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.bn2.weight.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.bn3.weight.requires_grad = False
        self.fc1.weight.requires_grad = False

    def unfreeze(self):
        self.conv1.weight.requires_grad = True
        self.bn1.weight.requires_grad = True
        self.conv2.weight.requires_grad = True
        self.bn2.weight.requires_grad = True
        self.conv3.weight.requires_grad = True
        self.bn3.weight.requires_grad = True
        self.fc1.weight.requires_grad = True

    def reset_head(self, outputs: int = None):
        if not outputs:
            self.head.reset_parameters()
        else:
            self.head = nn.Linear(512, outputs)
