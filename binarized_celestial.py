import torch
import torch.nn as nn
import math
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd.function  import Function, InplaceFunction

class binarize(InplaceFunction):

    def forward(ctx,input,quant_mode='det',allow_scale=False,inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()      

        scale= output.abs().max() if allow_scale else 1

        if quant_mode=='det':
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1).mul(scale)
        
    def backward(ctx,grad_output):
        #STE 
        grad_input=grad_output
        return grad_input,None,None,None

def binarized(input,quant_mode='det'):
      return binarize.apply(input,quant_mode)  

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input_b=binarized(input)
        weight_b=binarized(self.weight)
        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
    
class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b=input
        weight_b=binarized(self.weight)

        out = nn.functional.conv2d(input_b, weight_b, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

class binarized_celestial(nn.Module):

    def __init__(self, num_classes=6):
        super(binarized_celestial, self).__init__()
        self.ratioInfl = 3
        self.features = nn.Sequential(
            BinarizeConv2d(3, int(64*self.ratioInfl), kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(64*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
            BinarizeConv2d(int(64*self.ratioInfl), int(192*self.ratioInfl), kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(int(192*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(192*self.ratioInfl), int(384*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(384*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(384*self.ratioInfl), int(256*self.ratioInfl), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(256*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(256*self.ratioInfl), 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True)
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear(4096, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.LogSoftmax()
        )
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            20: {'lr': 1e-3},
            30: {'lr': 5e-4},
            35: {'lr': 1e-4},
            40: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
def celestial_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 6)
    return binarized_celestial(num_classes)