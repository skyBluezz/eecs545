import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

criterion = nn.CrossEntropyLoss()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cal_importance_fisher(net, l_id, num_stop=100):
    optimizer = optim.SGD(net.parameters(), lr=0, weight_decay=0)
    num = 0
    bias_base = l_id.bias.data.clone().detach()
    imp_corr_bn = torch.zeros(bias_base.shape[0]).to(device)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        imp_corr_bn += (((l_id.weight.grad)*(l_id.weight.data)) + ((l_id.bias.grad)*(l_id.bias.data))).pow(2)
        num += labels.shape[0]
        if(num > num_stop):
            break
    
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    
    return neuron_order


def cal_importance_tfo(net, l_id, num_stop=100):
    optimizer = optim.SGD(net.parameters(), lr=0, weight_decay=0)
    num = 0
    bias_base = l_id.bias.data.clone().detach()
    imp_corr_bn = torch.zeros(bias_base.shape[0]).to(device)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        imp_corr_bn += (((l_id.weight.grad)*(l_id.weight.data)) + ((l_id.bias.grad)*(l_id.bias.data))).abs()
        num += labels.shape[0]
        if(num > num_stop):
            break
    
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    
    return neuron_order


def cal_importance_netslim(net, l_id, num_stop=100):
    imp_corr_bn = l_id.weight.data.abs()    
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    return neuron_order


def cal_importance_l1(net, l_id, num_stop=100):
    imp_corr_bn = l_id.weight.data.abs().sum(dim=(1,2,3))
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    return neuron_order

def cal_importance_l2(net, l_id, num_stop=100):
    imp_corr_bn = l_id.weight.data.pow(2).sum(dim=(1,2,3))
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    return neuron_order

def cal_importance_rd(net, l_id, num_stop=100, num_of_classes=100):
    optimizer = optim.SGD(net.parameters(), lr=0, weight_decay=0)
    sm = nn.Softmax(dim=1)
    lsm = nn.LogSoftmax(dim=1)
    num = 0
    bias_base = l_id.bias.data.clone().detach()
    imp_corr_bn = torch.zeros(bias_base.shape[0]).to(device)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        prob, log_prob = sm(outputs).mean(dim=0), lsm(outputs).mean(dim=0)

        for j in range(num_of_classes):
            optimizer.zero_grad()
            log_prob[j].backward(retain_graph=True)
          
            imp_corr_bn += prob[j].item()*l_id.weight.grad.data**2

        num += labels.shape[0]
        if(num > num_stop):
            break
    
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    
    return neuron_order
