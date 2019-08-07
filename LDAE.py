import numpy as np
import json
import os.path
from uuid import uuid1
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torchsummary import summary
from time import gmtime, strftime
import time
import copy
from gagdatase_keras import  DataGenerator
from model.configuration import Configuration
from model.encoder import randomize_users


#batch of true age
y_true = np.array([10,20,50,75])
#convert to torch
y_true = torch.from_numpy(y_true) 
y_true = y_true.view(-1, 1)
# predicted age
from torch.distributions import uniform
distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([10.0]))
y_pred=distribution.sample(torch.Size([400]))
y_pred=torch.reshape(y_pred, (-1, 100)) # matrix of size[4,100]


sigma = torch.FloatTensor([2.5])
denom = torch.FloatTensor([np.sqrt(2 * np.pi)])
cdx = torch.zeros((4, 100), dtype=torch.float)
cdx[:, :] += torch.arange(100, dtype=torch.float)

soft_max = torch.softmax(y_pred, dim=1).clamp(min=1e-8) # softmax shape [32, 100]

soft_weights = cdx - y_true.float() # [32, 100]   , batch_age.view(-1, 1) == [32, 1]
targets = torch.exp(-torch.pow(soft_weights, 2) / (sigma * sigma)) / (sigma * denom) # [32, 100]

hard_losses = torch.sum(targets * torch.log(soft_max), dim=1) #[32]
soft_losses = torch.sum((1 - targets) * torch.log((1 - soft_max).clamp(min=1e-8)), dim=1)# [32]
LDAE = - torch.sum((hard_losses + soft_losses)) / 4
