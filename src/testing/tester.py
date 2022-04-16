import os
import cv2
import tqdm
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Tester():
    def __init__(self, G, test_dir, outputs_dir, device):
        
        self.G = G
        self.test_dir = test_dir
        self.outputs_dir = outputs_dir
        self.device = device

    def generate_example(self,input_image):
        input_image = cv2.resize(input_image,(512,512))
        input_image=np.transpose(input_image,(2,0,1))
        input_image = (input_image - 127.5) / 127.5
        input_image = input_image.astype(np.float32)
        x = torch.tensor(input_image).to(self.device)
        x = x.unsqueeze(dim=0)
        fake_image = self.G(x)[3]
        self.G.zero_grad()
        input_image = (np.transpose(input_image,(1,2,0))+1)/2
        fake_image = np.transpose(np.array(fake_image.detach().cpu())[0],(1,2,0))
        fake_image = (fake_image+1)/2
        output_image = np.concatenate([input_image,fake_image],axis=1)
        return output_image
        
    def fit(self):
        l_images = os.listdir(self.test_dir)
        for file_name in tqdm.tqdm(l_images):
            image_path = os.path.join(self.test_dir,file_name)
            input_image = np.array(Image.open(image_path))
            output_image = self.generate_example(input_image=input_image)
            out_path = os.path.join(self.outputs_dir,file_name)
            plt.imsave(out_path,output_image)
        return None
        

