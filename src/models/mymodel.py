import torch
from torch import nn
# Define model

def conv33(Cin,Cout):
    return nn.Conv2d(in_channels=Cin, out_channels=Cout, kernel_size=3, stride=1, padding=1, bias=False)
def conv3d(Cin,Cout,ksize=(3,3,3),stride=(1,1,1),padding=(1,1,1)):    
    return nn.Conv3d(in_channels=Cin, out_channels=Cout, kernel_size=ksize, stride=stride, padding=padding, bias=False)
def bn(Cin):
    return nn.BatchNorm2d(num_features=Cin)
def bn3d(Cin):
    return nn.BatchNorm3d(num_features=Cin)
def relu():
    return nn.ReLU()
def pool():
    return nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
def pool3d():
    return nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
def pool2d():
    return nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
    

def BRC(Cin,Cout):
    return nn.Sequential(
        bn(Cin),
        relu(),
        conv33(Cin,Cout)
    )

def BRC3d(Cin,Cout,stride=1):
    return nn.Sequential(
        bn3d(Cin),
        relu(),
        conv3d(Cin,Cout),
        bn3d(Cout),
        relu(),
        conv3d(Cout,Cout,stride=stride)
    )

def BRC2d(Cin,Cout,stride=1):
    return nn.Sequential(
        bn3d(Cin),
        relu(),
        conv3d(Cin,Cout,ksize=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
        bn3d(Cout),
        relu(),
        conv3d(Cout,Cout,ksize=(1,3,3),stride=stride,padding=(0,1,1))
    )
    

class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        ''' entry '''
        # NDHWC : (N,8,128,128,1) -> (N,8,64,64,32)
        self.entry_01_conv = conv3d(1,32,(1,5,5),(1,2,2),(0,2,2)) 
        
        ''' encoder '''
        self.enc_01_BRC = BRC2d(32,32,stride=(1,2,2))
        # (N,8,32,32,32)
        
        self.enc_02_BRC = BRC3d(32,64,stride=(2,2,2))
        # (N,4,16,16,64)
        
        self.enc_03_BRC = BRC2d(64,128,stride=(1,2,2))
        # (N,4,8,8,,128)
        
        self.enc_04_BRC = BRC3d(128,256,stride=(2,2,2))
        # (N,2,4,4,256)
        
        
        ''' deep '''
        self.deep_01_BRC = BRC3d(256,256)
        self.deep_02_BRC = BRC3d(256,256)
        
        
        ''' decoder '''
        self.dec_01_resize = nn.Upsample(scale_factor=(2,4,4))
        # (N,4,16,16,256)
        self.dec_01_BRC_01 = BRC3d(256,64)
        self.dec_01_BRC_02 = BRC2d(64,64)
                
        self.dec_02_resize = nn.Upsample(scale_factor=(2,4,4))
        # (N,8,64,64,???)
        self.dec_02_BRC_01 = BRC2d(64+64,32)
        self.dec_02_BRC_02 = BRC2d(32,32)
        
        ''' exit '''
        self.exit_01_resize = nn.Upsample(scale_factor=(1,2,2))
        # (N,8,128,128,???)
        self.exit_01_conv = conv3d(64,2,(1,5,5),(1,1,1),(0,2,2))
            
        

    def forward(self, x):
        
        z = x.permute(0,4,1,2,3)
        
        ''' entry '''
        
        z = self.entry_01_conv(z)
        z_beg = z
        
        ''' encoder '''
        z = self.enc_01_BRC(z)
        z = self.enc_02_BRC(z)
        z_mid = z
        
        z = self.enc_03_BRC(z)
        z = self.enc_04_BRC(z)
        
        ''' deep '''
        z = self.deep_01_BRC(z)
        z = self.deep_02_BRC(z)
                
        ''' decoder '''
        z = self.dec_01_resize(z)
        z = self.dec_01_BRC_01(z)
        z = self.dec_01_BRC_02(z)
        
        z = torch.cat((z,z_mid),dim=1)
        z = self.dec_02_resize(z)
        z = self.dec_02_BRC_01(z)
        z = self.dec_02_BRC_02(z)
        
        ''' exit '''
        z = torch.cat((z,z_beg),dim=1)
        z = self.exit_01_resize(z)
        z = self.exit_01_conv(z)
        
        
        z = z.permute(0,2,3,4,1)
        
        return z
    
    
if __name__ == '__main__':
    
    import torchsummary
    
    model = mymodel()
    print(model)
    torchsummary.summary(model, (16,128,128,1), device='cpu')