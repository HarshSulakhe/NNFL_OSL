import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.utils import make_grid,save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

## CHANGE THIS TO ANY PATH OF AN IMAGE ON YOUR LOCAL SYSTEM
paths = ['/home/harsh/siamese-pytorch/omniglot/python/images_background/Tifinagh/character54/0963_05.png','/home/harsh/siamese-pytorch/omniglot/python/images_background/Cyrillic/character11/0228_06.png']

## TO CREATE TRANSFORMS WITHOUT MULTIDIMENSIONAL PROBABILITY DISTRIBUTION
## 2/105 in translate represents a shift of +- 2 pixels, with (105,105) being the original image resolution
data_transforms = transforms.Compose([
    transforms.RandomAffine(degrees = (-10,10),shear = (-.3,.3), scale = (0.8,1.2),fillcolor = 'white',translate = (0,2/105)),
    transforms.ToTensor()
])

## TO CREATE TRANSFORMS USING MULTIDIMENSIONAL PROBABILITY DISTRIBUTION
def create_transforms():
    deg = (-10,10) if np.random.randn() > 0.5 else (0,0)
    shear = (-.3,.3) if np.random.randn() > 0.5 else None
    scale = (0.8,1.2) if np.random.randn() > 0.5 else None
    translate = (0,2/105) if np.random.randn() > 0.5 else None
    return transforms.Compose([transforms.RandomAffine(deg,translate,scale,shear,fillcolor = 'white'),transforms.ToTensor()])

for path in paths:
    images = []
    img = Image.open(path).convert('L')
    # print(img.size)
    # print(type(img))
    for i in range(20):
        ## UNCOMMENT BELOW LINE AND COMMENT THE NEXT TO USE RANDOMNESS, DEFAULT IS WITHOUT RANDOMNESS FOR SHOWING RESULTS
        # img_ = create_transforms()(img)
        img_ = data_transforms(img)
        images.append(img_)
    grid = make_grid(images,padding = 0,nrow = 5)
    save_image(grid,'./AFFINE_'+path.split('/')[-3]+'_'+path.split('/')[-1])
