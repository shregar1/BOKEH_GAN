import torch
import torch.nn as nn

class res_block(nn.Module):
    def __init__(self,in_channels,out_channels,strides,identity_fn=None):
        super(res_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=strides,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.downsample=identity_fn
    def forward(self,x):
        identity=x
        x=self.conv1(x)
        x=self.relu(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.bn2(x)
        
        if self.downsample!=None:
            identity=self.downsample(identity)
        x=torch.add(x,identity)
        x=self.relu(x)
        return x    
            
class encoder(nn.Module):
    def __init__(self,res_block,layers,image_channels):
        super(encoder,self).__init__()
        self.conv=nn.Conv2d(image_channels,64, kernel_size=7, stride=2,padding=3,bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layers=layers
        self.layer1=self.res_layer(res_block,count=self.layers[0],in_channels=64,out_channels=64,stride=1)
        self.layer2=self.res_layer(res_block,count=self.layers[1],in_channels=64,out_channels=128,stride=2)
        self.layer3=self.res_layer(res_block,count=self.layers[2],in_channels=128,out_channels=256,stride=2)
        self.layer4=self.res_layer(res_block,count=self.layers[3],in_channels=256,out_channels=512,stride=2)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x_skip1=x
        x=self.maxpool(x)
        x=self.layer1(x)
        x_skip2=x
        x=self.layer2(x)
        x_skip3=x
        x=self.layer3(x)
        x_skip4=x
        x=self.layer4(x)
        return x,x_skip1,x_skip2,x_skip3,x_skip4
        
    def res_layer(self,res_block,count,in_channels,out_channels,stride=1):
        layers=[]
        if stride!=1:
            downsample=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=0,bias=False),
                                      nn.BatchNorm2d(out_channels))
            
        else:
            downsample=None
        layers.append(res_block(in_channels,out_channels,stride,downsample))
        
        for i in range(count-1):
            layers.append(res_block(out_channels,out_channels,1,None))
        return nn.Sequential(*layers)
            
        