
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import celestial
import test
import train
import read_dataset
import binarized_celestial
torch.serialization.add_safe_globals([celestial.CelestialNet])


DEVICE=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
EPOCH=10
BATCH_SIZE=32

train_loader = read_dataset.import_train_dataset(BATCH_SIZE)
test_loader = read_dataset.import_test_dataset(BATCH_SIZE)
valid_loader = read_dataset.import_valid_dataset(BATCH_SIZE)

#model = celestial.CelestialNet().to(DEVICE)
#model = models.resnet50(pretrained=True)
#model.fc = nn.Sequential(nn.Linear(model.fc.in_features,2),
                            #nn.LogSoftmax(dim=1))
#optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)
model = binarized_celestial.binarized_celestial().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9,weight_decay=0.0005)


list=[]
Train_Loss_list=[]
Valid_Loss_list=[]
Valid_Accuracy_list=[]

for epoch in range(1,EPOCH+1):
    #训练集训练
    train_loss=train.train_model(model,DEVICE,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    torch.save(model,r'./saveModel/model%s.pth'%epoch)
 
    #验证集进行验证
    test_loss,acc=test.test_model(model,DEVICE,valid_loader)
    Valid_Loss_list.append(test_loss)
    Valid_Accuracy_list.append(acc)
    list.append(test_loss)

min_num=min(list)
min_index=list.index(min_num)

print('model%s'%(min_index+1))
print('验证集最高准确率： ')
print('{}'.format(Valid_Accuracy_list[min_index]))

model=torch.load(r'./saveModel/model%s.pth'%(min_index+1), map_location=DEVICE, weights_only=False)
model.eval()
 
accuracy=test.test_model(model,DEVICE,test_loader)
print('测试集准确率')
print('{}%'.format(accuracy))