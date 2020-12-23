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

###########################
####DataDrivenPruning######
# 'Only considering Neurons in post-relu layer... (i.e, 192 in layer 6..)

# 'INPUTS: network, the array of indices for the relu layers, 
# 'RETURN: global importance, with shape below.  Sorted later by order_ratios'
###########################

def cal_importance_dataDriven(net, l_id, num_stop=100):
    print('Working 1')
    optimizer = optim.SGD(net.parameters(), lr=0, weight_decay=0)
    optimizer = optim.SGD(net.parameters(), lr=0, weight_decay=0)
    num = 0
    size = net.features[l_id-1].weight.shape[0]
    aPoZ = torch.zeros(size) #size of the conv layer before reLu...
 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        
        num += 1 
        if(num > num_stop):
            break
        print('Sample ', num)
    
        ##for neuron in post relu...
        ##if activation of weight is zero, add 1 to List at the weights index
        ##should this be before back propogation?
        l_id_output = net.features[0:l_id+1](inputs)  #output of the relu layer 
       # print(l_id_output.shape)

        for a in range(0,l_id_output.shape[1]):
               # print(l_id_output[:,a,:,:].shape)
                sum_ = 0
                sum_ = (0,np.count_nonzero(l_id_output.detach().numpy()[:,a,:,:] == 0))
                aPoZ[a] += sum_[1]*0.001
 
    imp_corr_bn = 1/aPoZ #so that entries with highest aPoZ have lowest importance
   # print(imp_corr_bn)
    neuron_order = [np.linspace(0, imp_corr_bn.shape[0]-1, imp_corr_bn.shape[0]), imp_corr_bn]
    #print('Neuron Order shape', len(neuron_order), len(neuron_order[1]))
    #index in layer, layer index, importance.... m x 3 array.
    
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
