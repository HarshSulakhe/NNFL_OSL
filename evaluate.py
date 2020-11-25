import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import time
import numpy as np
import gflags
import sys
from collections import deque
import os


if __name__ == '__main__':

    net = Siamese().cuda()
    net.eval()

    net.load_state_dict(torch.load('models/model-inter-151.pt'))

    testSet = OmniglotTest('omniglot/python/images_evaluation', transform=transforms.ToTensor(), times = 400, way = 20)
    testLoader = DataLoader(testSet, batch_size=20, shuffle=False)

    queue = deque(maxlen = 20)

    for i in range(20):
        right, error = 0, 0
        for _, (test1, test2) in enumerate(testLoader, 1):
            # if Flags.cuda:
            test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output = net.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                right += 1
            else: error += 1
        print('*'*70)
        print('\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%( right, error, right*1.0/(right+error)))
        print('*'*70)
        queue.append(right*1.0/(right+error))

    final = 0
    for acc in queue:
        final += acc

    print(final/20)
