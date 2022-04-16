import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.dataset import ImageDataset
from src.losses.loss import PerceptualLoss, ColorLoss, SSIMLoss, TotalVariationalLoss

class Trainer():
    def __init__(self,device,dataset_dir,outputs_dir, G, D, lambda_L1, lambda_Per, lambda_Col, lambda_Con, lambda_Sty, lambda_ssim, lambda_tv):
        
        #datloaders
        self.train_dataset = ImageDataset(root_dir=dataset_dir+"train",
                                          image_size=256+256)
        self.train_dataloader = DataLoader(self.train_dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=0)
        self.val_dataset = ImageDataset(root_dir=dataset_dir+"val",
                                        image_size=512)
        self.val_dataloader = DataLoader(self.val_dataset,batch_size=1,shuffle=True)

        self.G = G
        self.D = D
        
        #optimizers
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        #lambda
        self.lambda_L1 = lambda_L1
        self.lambda_Per = lambda_Per
        self.lambda_Col = lambda_Col
        self.lambda_ssim = lambda_ssim
        self.lambda_tv = lambda_tv
        
        #device
        self.device = device
        
        #output_dir
        self.output_dir = outputs_dir
        
        #losses
        self.BCE_Loss = nn.BCELoss().to(self.device)
        self.L1_Loss = nn.L1Loss().to(self.device)
        self.P_Loss = PerceptualLoss(self.device, lambda_Con, lambda_Sty)
        self.C_Loss = ColorLoss(self.device, 0, 0)
        self.SSIM_Loss = SSIMLoss()
        self.TV_Loss = TotalVariationalLoss()
    
    def compute_loss(self, y, y_fake):
        G_L1_loss = self.L1_Loss(y_fake,y)
            
        #Perceptual loss
        perceptual_loss = self.P_Loss.find(y,y,y_fake)
            
        #color loss
        color_loss = self.C_Loss.find(y,y_fake)
            
        #structural similarity loss
        ssim_Loss = self.SSIM_Loss.find(y,y_fake)
            
        #Total variational loss
        tv_loss = self.TV_Loss.find(y_fake)
        
        loss = self.lambda_L1*G_L1_loss + self.lambda_Per*perceptual_loss + self.lambda_Col*color_loss + self.lambda_ssim*ssim_Loss + self.lambda_tv*tv_loss
        
        return loss

    def train_fn(self):
        print(len(self.train_dataloader))
        for i,(input_image,target_image) in enumerate(tqdm.tqdm(self.train_dataloader)):
            x = input_image.to(self.device)
            y = target_image.to(self.device)
            
            #train_discriminator
            y_fake = self.G(x)
            D_real = self.D(x,y)
            D_fake = self.D(x,y_fake.detach())
            D_real_loss = self.BCE_loss(D_real,torch.ones_like(D_real))
            D_fake_loss = self.BCE_loss(D_fake,torch.zeros_like(D_fake))
            
            #Discriminator loss
            D_loss = (D_real_loss + D_fake_loss)/2
            
            self.D.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()
            
            #train_generator
            y_fake = self.G(x)
            D_fake = self.D(x,y_fake)
            G_fake_loss = self.BCE_Loss(D_fake,torch.ones_like(D_fake))
            
            #Generator loss
            G_loss = G_fake_loss + self.compute_loss(y,y_fake)
            
            self.G.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()
            
    def save_examples(self,epoch):
        print("saving")
        for j,(input_images,target_images) in enumerate(self.val_dataloader):
            x = input_images.to("cuda:0")
            fake_images = self.G(x)
            self.G.zero_grad()
            for i in range(len(fake_images)):
                input_image = (np.transpose(input_images[i],(1,2,0))+1)/2
                target_image = (np.transpose(target_images[i],(1,2,0))+1)/2
                fake_image = np.transpose(np.array(fake_images.detach().cpu())[i],(1,2,0))
                fake_image = (fake_image+1)/2
                output_image = np.concatenate([input_image,target_image,fake_image],axis=1)
                file_name = "image_epoch-"+str(epoch)+"_sample-"+str(i)+".png"
                file_path = os.path.join(self.output_dir,file_name)
                plt.imsave(file_path,output_image)
            break
        return
        
    def fit(self, num_epochs):
        self.G.to(self.device)
        self.D.to(self.device)
        self.G.train()
        self.D.train()
        
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []
        
        for epoch in range(num_epochs):
            print(epoch)
            #train
            self.train_fn(self.D,self.G)
            
            #checkpointing
            if epoch % 1 == 0:
                torch.save(self.G,"Generator_model_anime_2_sketch.pth")
                torch.save(self.D,"Discriminator_model_anime_2_sketch.pth")
            #save examples
            self.save_examples(epoch)
        

        