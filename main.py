
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import celestial
import test
import train
import read_dataset
torch.serialization.add_safe_globals([celestial.CelestialNet])


DEVICE=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
EPOCH=2
BATCH_SIZE=32

train_loader = read_dataset.import_train_dataset(BATCH_SIZE)
test_loader = read_dataset.import_test_dataset(BATCH_SIZE)
valid_loader = read_dataset.import_valid_dataset(BATCH_SIZE)

model = celestial.CelestialNet().to(DEVICE)
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0005)

list=[]
Train_Loss_list=[]
Valid_Loss_list=[]
Valid_Accuracy_list=[]

for epoch in range(1,EPOCH+1):
    #训练集训练
    train_loss=train.train_model(model,DEVICE,train_loader,optimizer,epoch)
    Train_Loss_list.append(train_loss)
    torch.save(model,r'.\saveModel\model%s.pth'%epoch)
 
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

model=torch.load(r'.\saveModel\model%s.pth'%(min_index+1), map_location=DEVICE, weights_only=False)
model.eval()
 
accuracy=test.test_model(model,DEVICE,test_loader)
print('测试集准确率')
print('{}%'.format(accuracy))
 
 
#绘图
#字体设置，字符显示
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
 
#坐标轴变量含义
x1=range(0,EPOCH)
y1=Train_Loss_list
y2=Valid_Loss_list
y3=Valid_Accuracy_list
 
#图表位置
plt.subplot(221)
#线条
plt.plot(x1,y1,'-o')
#坐标轴批注
plt.ylabel('训练集损失')
plt.xlabel('轮数')
 
plt.subplot(222)
plt.plot(x1,y2,'-o')
plt.ylabel('验证集损失')
plt.xlabel('轮数')
 
plt.subplot(212)
plt.plot(x1,y3,'-o')
plt.ylabel('验证集准确率')
plt.xlabel('轮数')
 
#显示
plt.show()