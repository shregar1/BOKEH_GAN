import os
import torch
import warnings
from training.trainer import Trainer
from models.resnet_generator import Net
from models.discriminator import Discriminator

warnings.filterwarnings("ignore",category=UserWarning)

train_stage = 1
dataset_dir = "dataset/train"
outputs_dir = "outputs/results/training"
weights_dir = "weights"
G_filename = None
D_finename = None
encoder_path = None
lambda_L1 = 100
lambda_Per = 1
lambda_Col = 50
lambda_Con = 1
lambda_Sty = 1
lambda_ssim = 1 
lambda_tv = 0.001
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Net(in_channels=3, out_channels=3, encoder_path=encoder_path)
D = Discriminator()

if G_filename is not None:
    G_path = os.path.join(weights_dir,G_filename)
    G = torch.load(G_path)
if D_finename is not None:
    D_path = os.path.join(weights_dir,D_finename)
    D = torch.load(D_path)

G.to(device)
D.to(device)

train = Trainer(device=device, dataset_dir=dataset_dir, outputs_dir=outputs_dir,
                G=G, D=D, lambda_L1=lambda_L1, lambda_Per=lambda_Per, lambda_Col=lambda_Col,
                lambda_Con=lambda_Con, lambda_Sty=lambda_Sty, lambda_ssim=lambda_ssim,
                lambda_tv=lambda_tv)
train.fit(num_epochs=num_epochs)

torch.save(train.G,f"Generator_model_dped_gan_stage_{train_stage}.pth")
torch.save(train.D,f"Discriminator_model_dped_gan_stage_{train_stage}.pth")