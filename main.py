import torch
import torch.nn as nn
import torch.nn.functional as F


import os
from PIL import Image
import numpy as np
data = {}
total = 0

def determine_janky_height(val):
  green_img = Image.open('data/green.png')
  green = np.array(green_img)[25, :750, :3]
  best = -1
  bestval = 10000000
  for x in range(750):
    distance = np.linalg.norm(green[x] - val)
    if distance < bestval:
      bestval = distance
      best = x
  return 25*best/750


def estimate_height_from_image(im):
  mean = np.mean(im, axis=(0,1))
  return determine_janky_height(mean)


x_train = np.zeros((36, 3, 400, 400))
y_train = np.zeros(36)
x_val = np.zeros((9, 3, 400, 400))
y_val = np.zeros(9)

x_test = np.zeros((3, 3, 400, 400))
y_test = np.zeros(3)
total = 0
for image in os.listdir('data/canopy_images'):
  if (".png" in image and ("0.png" not in image and int(image[:3]) > 400)):

    img = np.array(Image.open('data/satellite_images/' + image))[:, :, :3]
    target = np.array(Image.open('data/canopy_images/' + image))[:, :, :3]

    estimated_height = int(round(estimate_height_from_image(target)))
    #print(img.shape)
    img = np.moveaxis(img, 0, 2)
    img = np.moveaxis(img, 0, 1)
    target = np.moveaxis(target, 0, 2)
    target = np.moveaxis(target, 0, 1)
    #print(img.shape)
    if (total < 36):
      x_train[total] = img
      y_train[total] = estimated_height
    elif (total < 45):
      x_val[total - 36] = img
      y_val[total - 36] = estimated_height
    else:
      x_test[total - 45] = img
      y_test[total - 45] = estimated_height
    total += 1

    #()
  
data["X_train"] = x_train
data["y_train"] = y_train
data["X_val"] = x_val
data["y_val"] = y_val
data["X_test"] = x_test
data["y_test"] = y_test
print(np.max(y_train))
input()
"""
print(total)


# Load the (preprocessed) CIFAR-10 data.
data = get_CIFAR10_data()
for k, v in list(data.items()):
    print(f"{k}: {v.shape}")
    print(type(v))
    """



class Model(nn.Module):
 def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.conv2 = nn.Conv2d(20, 40, 5)
self.fc1 = nn.Linear(320, 10)

def forward(self, x):
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = x.view(-1, 320)
       x = F.relu(self.fc1(x))
       return F.log_softmax(x)

net = Model()