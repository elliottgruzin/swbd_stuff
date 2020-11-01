# -*- coding: utf-8 -*-
import torch
import pickle
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Check GPU available

use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True

# %% FFN Class

class ExternalNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, layer_width):
        self.num_layers = num_layers
        super(ExternalNetwork, self).__init__()
        self.f1 = nn.Linear(hidden_size*2, layer_width)
        self.f2 = nn.Linear(layer_width, layer_width)
        self.f3 = nn.Linear(layer_width, layer_width)
        self.f4 = nn.Linear(layer_width, layer_width)
        self.out = nn.Linear(layer_width, 1)

    def forward(self, in_data):
        'Can select from a number of diff layer combinations when model defined'
        o1 = F.relu(self.f1(in_data))
        if self.num_layers == 1:
            x = self.out(o1)
        if self.num_layers == 2:
            o2 = F.relu(self.f2(o1))
            x = self.out(o2)
        if self.num_layers == 3:
            o2 = F.relu(self.f2(o1))
            o3 = F.relu(self.f3(o2))
            x = self.out(o3)
        if self.num_layers == 4:
            o2 = F.relu(self.f2(o1))
            o3 = F.relu(self.f3(o2))
            o4 = F.relu(self.f4(o3))
            x = self.out(o4)
        return x
