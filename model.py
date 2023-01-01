# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 12:24:01 2023

@author: Khaled
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.sepConv1 = SeparableConv2d(in_channels, out_channels, 3)       
        self.sepConv2 = SeparableConv2d(out_channels, out_channels,3)
        self.conv = nn.Sequential(
            self.sepConv1, # We set bias to false because we use batch norm layer 3 is kernel size, stride is 1 and padding is 1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            self.sepConv2,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
    def forward(self,x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self,in_channels = 3, out_channels = 1, features=[64,128,256,512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList() # we use moduleList to do model.eval
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #Upsampling
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(DoubleConv(feature*2,feature)) #this is needed to go up then add 2 convs, refer to UNET design
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) #we want the last feature in feature list (512) and output 1028
        self.final_conv = nn.Conv2d(features[0],out_channels, kernel_size=1)
        
    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0,len(self.ups),2): #start from 0, within number of ups, increment by 2
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] #because we increment by 2
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])  #to make sure the input and output are of same size, we do [2:] to take height and width and skip batch size
                
            concat_skip = torch.cat((skip_connection,x),dim = 1) # to concat. skip connections
            x = self.ups[idx+1](concat_skip)
            
        return self.final_conv(x)
            

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    
if __name__ == "__main__":
    test()