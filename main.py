from cProfile import label
import enum
from dataset import CustomDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from typing import List
import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data_utils
from torch.autograd import Variable
from sklearn.metrics import balanced_accuracy_score
from models import myAutoEncoder
from torch.utils.data import DataLoader

mydataset = CustomDataset()
train_loader = DataLoader(dataset=mydataset, batch_size=2)

mymodel = myAutoEncoder()
print(mymodel)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mymodel.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0        
    #for x, y, z, labels in train_loader:
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        (x,y,z),labels = data
        labels = labels.squeeze()
        labels = labels.long()
        preds = mymodel(x.float(), y.float(), z.float())
        #preds = torch.argmax(preds, axis=1)
        print("PREDS--------", preds)
        
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
        

print("FINISHED TRAINING !")
