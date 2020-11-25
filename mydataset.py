import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image
from torchvision import transforms

class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        # agrees = [0, 90, 180, 270]
        idx = 0
        # for agree in agrees:
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading training dataset to memory")
        return datas, idx

    def create_transforms(self):
        deg = (-10,10) if np.random.randn() > 0.5 else (0,0)
        shear = (-.3,.3) if np.random.randn() > 0.5 else None
        scale = (0.8,1.2) if np.random.randn() > 0.5 else None
        translate = (0,2/105) if np.random.randn() > 0.5 else None
        return transforms.Compose([transforms.RandomAffine(deg,translate,scale,shear,fillcolor = 'white'),transforms.ToTensor()])

    def __len__(self):
        return  60000*128

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            t = self.create_transforms()
            image1 = t(image1)
            image2 = t(image2)
        else:
            image1 = transforms.ToTensor()(image1)
            image2 = transforms.ToTensor()(image2)

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=200, way=20):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath, charPath)):
                    filePath = os.path.join(dataPath, alphaPath, charPath, samplePath)
                    datas[idx].append(Image.open(filePath).convert('L'))
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def create_transforms(self):
        deg = (-10,10) if np.random.randn() > 0.5 else (0,0)
        shear = (-.3,.3) if np.random.randn() > 0.5 else None
        scale = (0.8,1.2) if np.random.randn() > 0.5 else None
        translate = (0,2/105) if np.random.randn() > 0.5 else None
        return transforms.Compose([transforms.RandomAffine(deg,translate,scale,shear,fillcolor = 'white'),transforms.ToTensor()])

    def __len__(self):
        # return  np.sum(list(len(self.datas[key]) for key in self.datas.keys()))
        return self.times*self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1 = random.choice(self.datas[self.c1])
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            t = self.create_transforms()
            img1 = t(self.img1)
            img2 = t(img2)

        else:
            img1 = transforms.ToTensor()(self.img1)
            img2 = transforms.ToTensor()(img2)


        return img1, img2


# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    print(omniglotTrain)
