import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from tqdm import tqdm
from PIL import Image
from . import seg_transforms

class CarvanaDataset(Dataset):
    
    def __init__(self,img_path,label_path,transform=None):
        super().__init__()
        
        # print( len(os.listdir(img_path)), "<--> ", len(os.listdir(label_path)))
        self.img_path = img_path
        self.mask_path = label_path
        self.sz = len(os.listdir(img_path))
        self.img_dir = os.listdir(img_path)
        
        # self.data = []
        self.transform = transform  
            
    def __len__(self):
        return self.sz
    
    def __getitem__(self, index):
        img_file = self.img_dir[index].split('/')[-1]
        mask_file = img_file.replace(".jpg","_mask.gif")
        # Open as image
        img = Image.open(self.img_path + img_file).convert('RGB')
        mask = Image.open(self.mask_path + mask_file).convert('L')
        # convert to np
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.uint8)
        data_item = (img,mask)
        data_item = self.transform(*data_item)
        return data_item


def load_dataset(img_path,label_path,batch_size=32,num_workers=1,transform=None):
    dataset = CarvanaDataset(img_path,label_path,transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

if __name__ == "__main__":
    pass