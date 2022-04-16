import os
import torch
from src.models.resnet_generator import Net
from src.models.discriminator import Discriminator
from src.training.trainer import Trainer

train_stage = 1
dataset_dir = "dataset"
outputs_dir = "outputs/results/training"
weights_dir = "weights"
G_filename = "Generator_model_dped.pth"
D_finename = "Discriminator_model_dped.pth"
lambda_L1 = 100
lambda_Per = 1
lambda_Col = 50
lambda_Con = 1
lambda_Sty = 1
lambda_ssim = 1 
lambda_tv = 1
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_path = os.path.join(weights_dir,G_filename)
D_path = os.path.join(weights_dir,D_finename)
G = Net(in_channels=3, out_channels=3, encoder_path=encoder_path)
D = Discriminator()
G.to(device)
D.to(device)

if G_path is not None:
    G = torch.load(G_path)
if D_path is not None:
    D = torch.load(D_path)

train = Trainer(device=device, dataset_dir=dataset_dir, outputs_dir=outputs_dir,
                G=G, D=D, lambda_L1=lambda_L1, lambda_Per=lambda_Per, lambda_Col=lambda_Col,
                lambda_Con=lambda_Con, lambda_Sty=lambda_Sty, lambda_ssim=lambda_ssim,
                lambda_tv=lambda_tv)
train.fit(num_epochs=num_epochs)

torch.save(train.G,f"Generator_model_dped_gan_stage_{train_stage}.pth")
torch.save(train.D,f"Discriminator_model_dped_gan_stage_{train_stage}.pth")