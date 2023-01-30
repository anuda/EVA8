import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR

def norm_layer(norm_type,channels=0,num_groups=0):
    assert channels != 0 or channels is not None, "Channels must be valid"
    
    if norm_type=='BN':
        norm_layer = nn.BatchNorm2d(channels)
    elif norm_type=='GN':
        norm_layer = nn.GroupNorm(num_groups, channels)

        
    return norm_layer

class Net(nn.Module):
    def __init__(self,norm_type='BN'):
        
        super(Net,self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding = 0, bias=False),
            norm_layer(norm_type,8),
            nn.ReLU()
            
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding = 0, bias=False),
            norm_layer(norm_type,16),
            nn.ReLU()
            
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3,3), padding = 0, bias=False),
            norm_layer(norm_type,20),
            nn.ReLU()
            
        )
        self.pool1 = nn.MaxPool2d(2,2)
        

        #create a transition 
        
        
        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,3), padding = 0, bias=False),
            norm_layer(norm_type,20),
            nn.ReLU()
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1,1), padding = 0, bias=False),
            norm_layer(norm_type,10),
            nn.ReLU()
        )
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding = 0, bias=False),
            norm_layer(norm_type,20),
            nn.ReLU()
        )
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3,3), padding = 0, bias=False),
            norm_layer(norm_type,10),
            nn.ReLU()
        )
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.1)
        
        
    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock7(x)
        
        x = self.convblock5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.gap(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x,dim=-1)
            