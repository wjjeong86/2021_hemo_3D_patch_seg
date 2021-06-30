'''
https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py
에서 복사해옴.

'''


import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

def conv3d(Cin,Cout,kernel_size=(5,5,5),stride=(1,1,1),padding=(2,2,2)):    
    ksize = kernel_size
    ksize1 = (1,) + ksize[1:3]
    ksize2 = ksize[0:1] + (1,1)
    pad1 = (0,) + padding[1:3]
    pad2 = padding[0:1] + (0,0)
    
    if False:
        ## conv3d
        layer = nn.Conv3d(Cin,Cout,kernel_size=ksize,stride=stride,padding=padding,bias=False)
    elif True:
        ## sep conv3d
        layer = nn.Sequential(
            nn.Conv3d(Cin,Cout,kernel_size=ksize1,stride=stride,padding=pad1,bias=False),
            nn.Conv3d(Cout,Cout,kernel_size=ksize2,stride=(1,1,1),padding=pad2,bias=False)
        )
    else:
        G = 8 if Cin>=32 else 1
        layer = nn.Sequential(
            nn.Conv3d(Cin,Cin,groups=G,kernel_size=ksize,stride=stride,padding=padding,bias=False),
            nn.Conv3d(Cin,Cout,kernel_size=1,stride=1,padding=0,bias=False)
        )
    
    return layer


class BRC(nn.Module):
    def __init__(self, Cin, Cout, ksize=(5,5,5), stride=(1,1,1), padding=(2,2,2)):
        super().__init__()        
        self.bn1 = nn.BatchNorm3d(Cin)
        self.relu1 = nn.GELU()
        self.conv1 = conv3d(Cin,Cout,kernel_size=ksize,stride=stride,padding=padding)
        
    def forward(self,x):
        out = self.conv1( self.relu1( self.bn1( x)))
        return out
        
def BRC_N(C, ksize=(5,5,5), stride=(1,1,1), padding=(2,2,2), repeat=1):
    layers = []
    for _ in range(repeat):
        layers.append( BRC(C,C))
    return nn.Sequential(*layers)

class InputTransition(nn.Module):
    def __init__(self, Cout):
        super().__init__()
        self.conv1 = conv3d(1, 16, kernel_size=(5,5,5), padding=(2,2,2))
    
    def forward(self, x):
        out = self.conv1(x)
        return out

class DownTransition(nn.Module):
    def __init__(self, Cin, repeat, dropout=False):
        super().__init__()
        Cout = Cin*2
        self.down_brc = BRC( Cin, Cout, ksize=(5,5,5), stride=(2,2,2), padding=(2,2,2))        
        self.drop = nn.Dropout3d() if dropout else passthrough
        self.brc_n = BRC_N(Cout, ksize=(5,5,5), padding=(2,2,2), repeat=repeat)
        
    def forward(self, x):
        down = self.down_brc(x)
        out = self.drop(down)
        out = self.brc_n(out)
        out = torch.add(out, down)
        return out

class UpTransition(nn.Module):
    def __init__(self, Cin, Cout, repeat, dropout=False):
        super().__init__()
        self.drop1 = nn.Dropout3d() if dropout else passthrough
        self.drop2 = nn.Dropout3d()
        self.up_bn = nn.BatchNorm3d(Cin)
        self.up_relu = nn.GELU()
        self.up_conv = nn.ConvTranspose3d( Cin, Cout//2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.brc_n = BRC_N(Cout, ksize=(5,5,5), padding=(2,2,2), repeat=repeat)
        
    def forward(self, x, skipx):
        out = self.drop1(x)
        skipout = self.drop2(skipx)
        
        out = self.up_conv( self.up_relu( self.up_bn(out)))
        xcat = torch.cat((out, skipout), 1)
        out = self.brc_n(xcat)
        out = torch.add(out, xcat)
        return out
        

class OutputTransition(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.brc = BRC(Cin,Cout=2, ksize=(5,5,5), padding=(2,2,2))
        
    def forward(self,x):
        out = self.brc(x)
        return out

class VNetV2(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super().__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        self.down_tr128 = DownTransition(64, 3, dropout=True)
        self.down_tr256 = DownTransition(128, 2, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        
        x = x.permute(0,4,1,2,3)
        x = x.contiguous()
        
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        
        out = out.permute(0,2,3,4,1)
        out = out.contiguous()
        
        return out
    
    
    
        
if __name__ == '__main__':
    
    import numpy as np
    import torchsummary
    
    model = VNetV2().to('cpu')
    torchsummary.summary(model,(16,128,128,1), batch_size=-1, device='cpu') #NDHWC
