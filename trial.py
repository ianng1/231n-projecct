from re import X
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from torch.utils.data import DataLoader,Dataset
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split

data = {}
total = 0


df=pd.read_csv("data/satellite_images.csv")
train_set,test_set=train_test_split(df,test_size=0.25)
img_folder = "data/satellite_images/"

class ImageDataset(Dataset):
  def __init__(self,csv,img_folder,transform):
    self.csv=csv
    self.transform=transform
    self.img_folder=img_folder
    
    self.image_names=self.csv[:]['id']
    self.labels=np.array(self.csv.drop(['id', 'height'], axis=1))
  
#The __len__ function returns the number of samples in our dataset.
  def __len__(self):
    return len(self.image_names)

  def __getitem__(self,index):
    
    image=cv2.imread(self.img_folder+self.image_names.iloc[index]+'.png')
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image=self.transform(image)
    targets=self.labels[index]
    
    sample = {'image': image,'labels':targets}

    return sample



train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor()])

test_transform =transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.ToTensor()])


train_dataset=ImageDataset(train_set,img_folder,train_transform)
test_dataset=ImageDataset(test_set,img_folder,test_transform)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4,
    shuffle=True
)

def imshow(inp, title=None):
    """imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


# Get a batch of training data
images = next(iter(train_dataloader))
print(torch.mean(images['image'][0, :, :, :]))
plt.imshow(images['image'][0, :, :, :].permute(1, 2, 0)  )
input()
# Make a grid from batch
output = torchvision.utils.make_grid(images['image'])

imshow(output)
