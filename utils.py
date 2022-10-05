#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:21:25 2022

@author: yu
"""


import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy import io
import scipy
import torch.nn as nn
import copy
import torchvision
import torch.nn.functional as F
from scipy import linalg


def shuffle_data_set(data_x,data_y):
    np.random.seed(1)
    index = np.arange(0,len(data_y))
    index = np.random.permutation(index)
    data_x = data_x[index,:]
    data_y = data_y[index]
    return data_x,data_y

def sub_set_task(label_list,sample_number):
    
    download_mnist = True
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',    
        train=True,  
        transform = torchvision.transforms.ToTensor(),                                                      
        download = download_mnist,
    )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    
    
    Train_x = train_data.data.type(torch.FloatTensor).view(-1,28*28)/255.
    Train_y = train_data.targets
    Test_x = test_data.test_data.type(torch.FloatTensor).view(-1,28*28)/255.   # shape from (2000, 28, 28) to (2000, 784), value in range(0,1)
    Test_y = test_data.targets
    
    Train_x,Train_y = shuffle_data_set(Train_x,Train_y)
    train_index = []

    count = 0
    for i in range(len(Train_x)):
        if Train_y[i] in label_list:
            train_index.append(i)
            count = count+1
        if count >= sample_number*len(label_list):
            break
    
    test_index = []
    for i in range(len(Test_x)):
        if Test_y[i] in label_list:
            test_index.append(i)

    
    
    Train_y = np.array(Train_y)
    train_x = Train_x[np.array(train_index),:]
    train_y = Train_y[np.array(train_index)]
    
    train_x = torch.tensor(train_x).type(torch.FloatTensor)
    train_y = torch.tensor(train_y).type(torch.LongTensor)

    Test_y = np.array(Test_y)
    test_x = Test_x[np.array(test_index),:]
    test_y = Test_y[np.array(test_index)]
    
    test_x = torch.tensor(test_x).type(torch.FloatTensor)
    test_y = torch.tensor(test_y).type(torch.LongTensor)
    
    train_x = train_x.view(-1,1,28,28)
    test_x = test_x.view(-1,1,28,28)
    
    return train_x,train_y,test_x,test_y

def predict_accuracy(model,data_x,data_y):
    
    pred = torch.max(model(data_x),1)[1].data.numpy()
    accuracy = np.mean(pred == data_y.data.numpy())
    
    return accuracy

def cal_loss(model,data_x,data_y):


    device = torch.device("cpu")

    model = model.to(device)
    
    loss_func = nn.CrossEntropyLoss()
    out_put = model(data_x)
    loss = loss_func(out_put,data_y)
    
    return loss.data.cpu().numpy()

def transform_matrix_to_array(para_list):
    para_holder = []
    for para in para_list:
        para_holder.append(para.data.clone().cpu().numpy().reshape(1,-1))
    para_array = np.hstack(para_holder)
    return para_array

def transform_array_to_matrix(model,layer_index,para_array):
    para_list = []
    start_point = 0
    for i in layer_index:
        weight = list(model.parameters())[i]
        num_weight = np.prod(weight.shape)
        para_matrix = para_array[0][start_point:num_weight+start_point].reshape(weight.shape)
        para_list.append(torch.tensor(para_matrix))
        start_point += num_weight
    return para_list


############## calculate the fisher information matrix
def cal_fisher_information(model, data_x, data_y, layer_index):

    # Initialize stuff
    torch_dataset = Data.TensorDataset(data_x, data_y)
    train_loader = Data.DataLoader(torch_dataset, batch_size=1, shuffle=False, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Main loop
    fisher_info_matrix = 0
    for b_x, b_y in train_loader:

        # Get ouput and Loss
        out = model(b_x)
        loss = nn.CrossEntropyLoss()(out, b_y)

        # Compute Gradient
        optimizer.zero_grad()
        loss.backward()
        
        # Store gradients
        layer_weight_grad = []
        for i in layer_index:
            layer_weight_grad.append(list(model.parameters())[i].grad)
        
        # Change size
        sample_grad = transform_matrix_to_array(layer_weight_grad)

        # Compute contribution to fisher info matrix
        fisher_info_matrix += np.dot(sample_grad.T, sample_grad)

    fisher_info_matrix = fisher_info_matrix / len(train_loader)
    
    # Get eigenvalues
    v,w = linalg.eig(fisher_info_matrix, )

    return fisher_info_matrix, np.real(v), np.real(w).T

######### calculate hessian matrix
def cal_hessian(model, data_x, data_y, layer_index):
    # Initialize stuff
    model = copy.deepcopy(model)
    loss_func = nn.CrossEntropyLoss()
    layer = layer_index[0]
    
    # calculate grad
    parameter = list(model.parameters())[layer]
    out_put = model(data_x)
    loss = loss_func(out_put,data_y)
    grad_1 = torch.autograd.grad(loss,parameter,create_graph=True)[0]
    
    # calculate hessian
    hessian = []
    for grad in grad_1.view(-1):
        grad_2 = torch.autograd.grad(grad,parameter,create_graph=True)[0].view(-1)
        hessian.append(grad_2.data.numpy())
    h = np.array(hessian)
    
    # get eigenvalues
    eigenvalue,eigenvector = np.linalg.eig(h)
    return hessian,np.real(eigenvalue),np.real(eigenvector).T
