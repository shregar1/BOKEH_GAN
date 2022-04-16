import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self,root_dir,image_size):
        self.root_dir=root_dir
        self.df_data = pd.read_csv(os.path.join(self.root_dir,"dataset.csv"))
        self.image_size=image_size
        
    def __len__(self):
        return len(self.list_files)
        
    def __getitem__(self,idx):
        input_file = self.df_data.iloc[idx]["input"]
        input_path = os.path.join(self.root_dir,input_file)
        input_image = np.array(Image.open(input_path))
        target_file = self.df_data.iloc[idx]["target"]
        target_path = os.path.join(self.root_dir,target_file)
        target_image = np.array(Image.open(target_path))
        input_image=cv2.resize(input_image,(self.image_size,self.image_size))
        target_image=cv2.resize(target_image,(self.image_size,self.image_size))
        input_image=np.transpose(input_image,(2,0,1))
        target_image=np.transpose(target_image,(2,0,1))
        input_image = (input_image - 127.5) / 127.5
        target_image = (target_image - 127.5) / 127.5
        input_image = input_image.astype(np.float32)
        target_image = target_image.astype(np.float32)
        return input_image,target_image
        