import torch
import torch.nn as nn
from models.modules.resnet import encoder
from models.modules.resnet import res_block
from models.modules.attention_unet import unet_block, out_block, skip_block

class Net(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_path=None):
        super(Net, self).__init__()
        self.encoder = encoder(res_block,[3,4,6,3],in_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.bottle_conv1=nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1)
        self.bottle_conv2=nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.unet_block3=unet_block(1024,256,256)
        self.out_3=out_block(in_channels=512,out_channels=out_channels)
        self.skip_3 = skip_block(in_channels=128, feature_map_channel=3)
        self.unet_block2=unet_block(512,128,128)
        self.out_2=out_block(in_channels=256,out_channels=out_channels)
        self.skip_2 = skip_block(in_channels=64, feature_map_channel=3)
        self.unet_block1=unet_block(256,64,64)
        self.out_1=out_block(in_channels=128,out_channels=out_channels)
        self.skip_1 = skip_block(in_channels=64, feature_map_channel=3)
        self.unet_block=unet_block(128,64,64)
        self.conv_trans=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0)
        self.out=nn.Conv2d(in_channels=67,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        x_ = x
        x,skip1,skip2,skip3,skip=self.encoder(x)
        x=self.bottle_conv1(x)
        x=self.relu(x)
        x=self.bottle_conv2(x)
        x=self.relu(x)
        x=self.unet_block3(x,skip)
        out_3 = self.out_3(x)
        skip3 = self.skip_3(skip3,out_3)
        x=self.unet_block2(x,skip3)
        out_2 = self.out_2(x)
        skip2 = self.skip_2(skip2,out_2)
        x=self.unet_block1(x,skip2)
        out_1 = self.out_1(x)
        skip1 = self.skip_1(skip1,out_1)
        x=self.unet_block(x,skip1)
        x=self.conv_trans(x)
        x=torch.cat([x_,x], dim=1)
        y=self.out(x)
        return out_3, out_2, out_1, y
