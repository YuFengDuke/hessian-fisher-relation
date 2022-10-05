#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:25:54 2022

@author: yu
"""


import matplotlib.pyplot as plt
from utils import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class FC(nn.Module):
    def __init__(self,H):
        super(FC,self).__init__()
        self.net = nn.Sequential(
                Flatten(),
                nn.Linear(784,H,bias = False),
                nn.ReLU(),
                nn.Linear(H,H,bias =False),
                nn.ReLU(),
                nn.Linear(H,10,bias = False),
                )
    def forward(self,x):
        output = self.net(x)
        return output



# digits to include
sample_holder = [0,1,2,3,4,5,6,7,8,9]

# leanring rate
LR = 0.1

# batch size
batch_size = 50

# max epoch
EPOCH = 2000

# size of test data
test_size = 1000

# number of data per digits, total training size is smaple_num * len(smaple_holder)
sample_num  = 1000

# the index for parameter, start from 0.
layer_index = [1]

# network size
net_size = 30

# the stop condition when loss below this value
stop_loss = 1e-4

# whether calculate the fisher and hessian
evaluation = True

# list to record training accuracy
train_accuracy_holder = []

# prepare data, sample_hodler: the included digits; smaple num: number of sample per digits
train_x,train_y,test_x,test_y = sub_set_task(sample_holder,sample_num)

# initilize model and optimizer
model = FC(net_size)
optimizer = torch.optim.SGD(model.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

# load data
torch_dataset = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(torch_dataset, batch_size = batch_size, shuffle=True)


for epoch in range(EPOCH):
    model.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        # get output
        out_put = model(b_x)
        
        # calculate loss
        loss = loss_func(out_put,b_y)
        
        # reset grad
        optimizer.zero_grad()
        
        # backward grad
        loss.backward()
        
        # update 
        optimizer.step()

        
    model.eval()
    
    train_loss = cal_loss(model, train_x, train_y)
    train_accuracy = predict_accuracy(model,data_x = train_x,data_y = train_y)
    train_accuracy_holder.append(train_accuracy)
    
    if (epoch%1 == 0):
        print('Epoch is |',epoch,
              'train accuracy is |',train_accuracy,
              'train loss is |', train_loss)
        
    if (np.mean(train_accuracy_holder[-1:-10:-1])>0.99)&(train_loss < stop_loss):
        break


if evaluation == True:
    model.eval()
    hessian_matrix,hessian,hessian_vector = cal_hessian(
        model,
        data_x = train_x,
        data_y = train_y,
        layer_index = layer_index
        )

    fisher_matrix,fisher,fisher_vector = cal_fisher_information(
        model,
        data_x = train_x,
        data_y = train_y,
        layer_index = layer_index
        )
    
    plt.figure()
    plt.plot(hessian, fisher, 's')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('H_i')
    plt.ylabel('F_i')