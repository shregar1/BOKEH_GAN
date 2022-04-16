import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               padding_mode="reflect")
        
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False,
                               padding_mode="reflect")
        self.norm2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False,
                               padding_mode="reflect")
        self.norm3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=4,
                               stride=1,
                               padding=1,
                               bias=False,
                               padding_mode="reflect")
        self.norm4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(in_channels=512,
                               out_channels=1,
                               kernel_size=4,
                               stride=1,
                               padding=1,
                               padding_mode="reflect")
        
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.lrelu(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.lrelu(x)
        x=self.conv3(x)
        x=self.norm3(x)
        x=self.lrelu(x)
        x=self.conv4(x)
        x=self.norm4(x)
        x=self.lrelu(x)
        x=self.conv5(x)
        x=torch.sigmoid(x)
        return x

