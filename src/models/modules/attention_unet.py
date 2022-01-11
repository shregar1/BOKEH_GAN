import torch
import torch.nn as nn
from fastai.layers import PixelShuffle_ICNR, in_channels

class attention(nn.Module):
    def __init__(self,in_channels):
        super(attention,self).__init__()
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels,1,kernel_size=1,stride=1,padding=0)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x,g):
        x_=torch.add(x,g)
        x_=self.relu(x)
        x_=self.conv(x)
        x_=self.sigmoid(x)
        x=torch.mul(x,x_)
        return x
    
class out_block(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(out_block,self).__init__()
        self.out_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        out = self.out_conv(x)
        return out

class skip_block(nn.Module):
    def __init__(self, in_channels) -> None:
        super(skip_block,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_channels=in_channels+1, out_channels=in_channels, kernel_size=1, stride=1)
    
    def forward(self,skip, y_):
        y_=self.upsample(y_)
        skip=torch.cat([skip,y_],dim=1)
        skip=self.conv(skip)
        return skip
        
class unet_block(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels):
        super(unet_block,self).__init__()
        final_channel=out_channels+skip_channels
        self.Pixelshuf=PixelShuffle_ICNR(in_channels,out_channels)
        self.attention=attention(out_channels)
        self.up_conv1=nn.Conv2d(final_channel,final_channel,kernel_size=3,stride=1,padding=1)
        self.up_conv2=nn.Conv2d(final_channel,final_channel,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        
    def forward(self,x,skip):
        x=self.Pixelshuf(x)
        skip=self.attention(skip,x)
        x=torch.cat([x, skip],dim=1)
        x=self.up_conv1(x)
        x=self.relu(x)
        x=self.up_conv2(x)
        x=self.relu(x)
        return x