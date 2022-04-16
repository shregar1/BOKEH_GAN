import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self,root_dir,image_size, d_type=None):
        self.root_dir=root_dir
        if d_type == "train":
            self.l_original_files = os.listdir(os.path.join(root_dir,"original"))[0:4000]
        else:
            self.l_original_files = os.listdir(os.path.join(root_dir,"original"))[4000:]
        self.df_data = pd.DataFrame({"input": self.l_original_files,
                                     "target": self.l_original_files})
        self.l_original_files.sort()
        self.image_size=image_size
        
    def __len__(self):
        return len(self.l_original_files)
        
    def __getitem__(self,idx):
        input_file = self.df_data.iloc[idx]["input"]
        input_path = os.path.join(self.root_dir,"original",input_file)
        input_image = np.array(Image.open(input_path))
        target_file = self.df_data.iloc[idx]["target"]
        target_path = os.path.join(self.root_dir,"bokeh",target_file)
        target_image = np.array(Image.open(target_path))
        input_image=cv2.resize(input_image,(self.image_size,self.image_size))
        target_image=cv2.resize(target_image,(self.image_size,self.image_size))
        target_image_3=cv2.resize(target_image,(self.image_size//4,self.image_size//4))
        target_image_2=cv2.resize(target_image,(self.image_size//8,self.image_size//8))
        target_image_1=cv2.resize(target_image,(self.image_size//16,self.image_size//16))
        input_image=np.transpose(input_image,(2,0,1))
        target_image_1=np.transpose(target_image_1,(2,0,1))
        target_image_2=np.transpose(target_image_2,(2,0,1))
        target_image_3=np.transpose(target_image_3,(2,0,1))
        target_image=np.transpose(target_image,(2,0,1))
        input_image = (input_image - 127.5) / 127.5
        target_image = (target_image - 127.5) / 127.5
        input_image = input_image.astype(np.float32)
        target_image_1 = target_image_1.astype(np.float32)
        target_image_2 = target_image_2.astype(np.float32)
        target_image_3 = target_image_3.astype(np.float32)
        target_image = target_image.astype(np.float32)
        return input_image,[target_image_1, target_image_2, target_image_3, target_image]
        