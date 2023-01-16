import torch
import torch.utils.data as data
import h5py
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms

#Note: Ensure data_rechunk.py has been run before creating instance of MTLDataset

class MTLDataset(data.Dataset):
    def __init__(self, type = 'train', transform=None):
        super().__init__()
        self.type = type
        self.transform = transform 

        with h5py.File(f'datasets/data_new/{self.type}/images.h5', 'r') as imgs:
            self.length = len(imgs['images'])   

    def __getitem__(self, index):

        with h5py.File(f'datasets/data_new/{self.type}/images.h5', 'r') as images:
            image_dset = images['chunked_images']
            image = image_dset[index]
            
        with h5py.File(f'datasets/data_new/{self.type}/bboxes.h5', 'r') as bboxes:
            bbox_dset = bboxes['chunked_bboxes']
            bbox = bbox_dset[index]

        with h5py.File(f'datasets/data_new/{self.type}/binary.h5', 'r') as binary:
            bin_dset = binary['chunked_binary']
            bin = bin_dset[index]

        with h5py.File(f'datasets/data_new/{self.type}/masks.h5', 'r') as masks:
            mask_dset = masks['chunked_masks']
            mask = mask_dset[index]
        
        #Ensuring binary target is an int
        if bin > 0:
            bin = 1
        else:
            bin = 0

        #Re-scaling pixel RGB values
        image = image / 255

        #Re-scaling bbox co-ordinates to the range [0,1]
        bbox = bbox / 255

        if self.transform:
            image = self.transform(image)
            #Normalizing images based on computed mean and std for Oxford Pets dataaset
            image = transforms.Normalize(mean=[0.4773, 0.4458, 0.3949],
             std=[0.2280, 0.2255, 0.2280])(image)
            mask = self.transform(mask)
        
        return image, bbox, bin, mask

    def __len__(self):
        
        return self.length