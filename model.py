import torch.nn as nn
import torch.nn.functional as F
import torch
import cal_utils as utils
from torch.autograd import Variable
import math
import numpy as np
 
class Data_Net(nn.Module):
    def __init__(self, input_dim=28*28, out_dim=20):
        super(Data_Net, self).__init__()
        mid_num1 = 2048
        mid_num2 = 1024
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3s = nn.Linear(mid_num2, out_dim, bias=False)
        nn.init.uniform_( self.fc3s.weight, -1. / np.sqrt(np.float(input_dim)), 1. / np.sqrt(np.float(input_dim)) )


        self.fc3e = nn.Linear(mid_num2, out_dim, bias=False)
        nn.init.uniform_( self.fc3e.weight, -1. / np.sqrt(np.float(input_dim)), 1. / np.sqrt(np.float(input_dim)) )


    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))

        out3e = self.fc3e(out2)
        norm_xe = torch.norm(out3e, p=2, dim=1, keepdim=True)
        out3e = out3e / norm_xe

        out3s = self.fc3s(out2)
        norm_xs = torch.norm(out3s, p=2, dim=1, keepdim=True)
        out3s = out3s / norm_xs

        return [out1, out2, out3e, out3s]

class Lab_Net(nn.Module):
    def __init__(self, input_dim=28*28, out_dim=20):
        super(Lab_Net, self).__init__()
        mid_num1 = 1024
        mid_num2 = 1024
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)
        self.fc3 = nn.Linear(mid_num2, out_dim)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)

        return [out1, out2, out3]

