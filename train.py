import torch
from torch import nn
from torch.nn import functional as F

def train_model(model,device,train_loader,optimizer,epoch):
    train_loss=0
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,label)
        loss.backward()
        optimizer.step()
        if batch_index%300==0:
            train_loss=loss.item()
            print('Train Epoch:{}\ttrain loss:{:.6f}'.format(epoch,loss.item()))

    return  train_loss