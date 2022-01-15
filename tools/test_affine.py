import numpy as np
import torch
from torch.nn import functional as F
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

img_path = './data/mpii/images/038641556.jpg'
img_torch = transforms.ToTensor()(Image.open(img_path))

theta = torch.tensor([
    [1,0,-0.2],
    [0,1,-0.5]
], dtype=torch.float)
grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
output = F.grid_sample(img_torch.unsqueeze(0), grid)
new_img_torch = output[0]
plt.imshow(new_img_torch.numpy().transpose(1,2,0))
plt.savefig('1.jpg')