import torch
import kornia
import torch.nn as nn
import torchvision.models as models

vgg_model = models.vgg16(pretrained=True)

for param in vgg_model.features.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg_model.cuda()

class LossModel(nn.Module):
    def __init__(self):
        super(LossModel,self).__init__()
        self.vgg_layers = vgg_model.features
        self.required_layers = {'3': "relu1_2",'8': "relu2_2",'15': "relu3_3",'22': "relu4_3"}
    def forward(self,x):
        output_activations = {}
        for name,module in self.vgg_layers._modules.items():
            x=module(x)
            if name in self.required_layers:
                output_activations.update({self.required_layers[name]:x})
        return output_activations
                
class PerceptualLoss():
    def __init__(self,device, lambda_Con, lambda_Sty):
        self.device = device
        self.lambda_Con = lambda_Con
        self.lambda_Sty = lambda_Sty
        self.loss_model=LossModel()
        for param in self.loss_model.parameters():
            param.requires_grad = False
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        
    def Gram_matrix(self,x):
        b,c,h,w = x.size()
        x = x.view(b,c,h*w)
        x_t = x.transpose(1,2)
        gram_matrix = x.bmm(x_t)
        return gram_matrix/(c*h*w)
        
    def fr_loss(self,out_content,out_target):
        total_feature_reconstruction_loss = self.L1_loss(out_target["relu3_3"],out_content["relu3_3"])
        return total_feature_reconstruction_loss
    
    def sr_loss(self,out_style,out_target):
        total_style_reconstruction_loss = 0
        for layer in out_style:
            gram_style = self.Gram_matrix(out_style[layer])
            gram_target = self.Gram_matrix(out_target[layer])
            total_style_reconstruction_loss += self.MSE_loss(gram_style,gram_target)
        return total_style_reconstruction_loss
            
        
    def find(self,x_content,x_style,y_target):
        out_content = self.loss_model.forward(x_content)
        out_style = self.loss_model.forward(x_style)
        out_target = self.loss_model.forward(y_target)
        
        #Feature Reconstruction Loss
        feature_reconstruction_loss = self.fr_loss(out_content,out_target)
        
        #Style Reconstruction Loss
        style_reconstruction_loss = self.sr_loss(out_style,out_target)
        
        perceptual_loss = self.lambda_Con*feature_reconstruction_loss + self.lambda_Sty*style_reconstruction_loss
        
        return perceptual_loss
        
class ColorLoss():
    def __init__(self,device, lambda_HSV, lambda_YUV):
        self.L1_loss = nn.L1Loss().to(device)
        self.lambda_HSV = lambda_HSV
        self.lambda_YUV = lambda_YUV
        self.Huber_loss = nn.SmoothL1Loss().to(device)
        
    def GB_loss(self,y_true,y_pred):
        y_true = kornia.filters.gaussian_blur2d(y_true)
        y_pred = kornia.filters.gaussian_blur2d(y_pred)
        gb_loss = self.L1_loss(y_true,y_pred)
        return gb_loss 
        
    def YUV_loss(self,y_true,y_pred):
        y_true = kornia.color.rgb_to_yuv(y_true)
        y_pred = kornia.color.rgb_to_yuv(y_pred)
        Y_loss = self.L1_loss(y_true[:,0,:,:],y_pred[:,0,:,:])
        U_loss = self.Huber_loss(y_true[:,1,:,:],y_pred[:,1,:,:])
        V_loss = self.Huber_loss(y_true[:,2,:,:],y_pred[:,2,:,:])
        YUV_loss = Y_loss + U_loss + V_loss
        return YUV_loss
    
    def HSV_loss(self,y_true,y_pred):
        y_true = kornia.color.rgb_to_hsv(y_true)
        y_pred = kornia.color.rgb_to_hsv(y_pred)
        H_loss = self.L1_loss(y_true[:,0,:,:],y_pred[:,0,:,:])
        S_loss = self.Huber_loss(y_true[:,1,:,:],y_pred[:,1,:,:])
        V_loss = self.Huber_loss(y_true[:,2,:,:],y_pred[:,2,:,:])
        HSV_loss = H_loss + S_loss + V_loss
        return HSV_loss*0.1
    
    def find(self,y_true,y_pred):
        y_true=(y_true+1)/2
        y_pred=(y_pred+1)/2
        color_loss = self.lambda_YUV*self.YUV_loss(y_true,y_pred) + self.lambda_HSV*self.HSV_loss(y_true,y_pred)
        return color_loss

class SSIMLoss():
    def __init__(self) -> None:
        pass
        
    def find(self, y_true, y_pred, window_size):
        ssim_loss = kornia.losses.ssim_loss(y_true,y_pred, window_size)
        return ssim_loss

class TotalVariationalLoss():
    def __init__(self) -> None:
        pass
    
    def find(self, y_pred):
        tv_loss = kornia.losses.total_variation(y_pred)
        tv_loss = torch.sum(tv_loss)
        return tv_loss

    
        