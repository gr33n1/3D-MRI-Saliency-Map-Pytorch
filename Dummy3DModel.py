import torch
from torch import nn
import torch.nn.functional as F


class Dummy3DModel(nn.Module):
    def __init__(self):
        super(Dummy3DModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3))
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.fc = nn.Linear(64 * 20 * 24 * 20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
