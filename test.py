import torch
from torch.nn import functional as F

def test_model(model,device,test_loader):
    model.eval()
    correct=0.0
    test_loss=0.0
 
    #不需要梯度的记录
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            test_loss+=F.cross_entropy(output,label).item()
            pred=output.argmax(dim=1)
            correct+=pred.eq(label.view_as(pred)).sum().item()
        test_loss/=len(test_loader.dataset)
        print('Test_average_loss:{:.4f},Accuracy:{:3f}\n'.format(
            test_loss,100*correct/len(test_loader.dataset)
        ))
        acc=100*correct/len(test_loader.dataset)
 
        return test_loss,acc
