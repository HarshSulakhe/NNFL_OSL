import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
import torch.nn as nn
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

    Flags = gflags.FLAGS
    gflags.DEFINE_bool("cuda", True, "use cuda")
    gflags.DEFINE_string("train_path", "omniglot/python/images_background", "training folder")
    gflags.DEFINE_string("test_path", "omniglot/python/images_evaluation", 'path of testing folder')
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 128, "number of batch size")
    gflags.DEFINE_float("lr", 0.0005, "learning rate")
    gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    gflags.DEFINE_integer("save_every", 50000, "save model after each save_every iter.")
    gflags.DEFINE_integer("test_every", 200, "test model after each test_every iter.")
    gflags.DEFINE_integer("max_iter", 50000, "number of iterations before stopping")
    gflags.DEFINE_string("model_path", "models", "path to store model")
    gflags.DEFINE_string("gpu_ids", "0  ", "gpu ids used to train")

    Flags(sys.argv)

    def weights_init(m):
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight.data,mean=0,std=0.01)
            nn.init.normal_(m.bias.data,mean=0.5,std=0.01)
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight.data,mean=0,std=0.2)
            nn.init.normal_(m.bias.data,mean=0.5,std=0.01)



    os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids


    trainSet = OmniglotTrain(Flags.train_path, transform=True)
    testSet = OmniglotTest(Flags.test_path, transform=False, times = Flags.times, way = Flags.way)
    testLoader = DataLoader(testSet, batch_size=Flags.way, shuffle=False, num_workers=Flags.workers)

    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    net = Siamese()

    if Flags.cuda:
        net.cuda()

    net.apply(weights_init)
    net.train()


    optimizer = torch.optim.SGD(net.parameters(),lr = Flags.lr,momentum=0.5)
    optimizer.zero_grad()
    lmbda = lambda epoch: 0.99
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lr_lambda = lmbda)


    train_loss = []
    loss_val = 0
    time_start = time.time()


    ## TRAINING LOOP FOR max_iter ITERATIONS
    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > Flags.max_iter:
            # print(batch_id)
            break
        if Flags.cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % Flags.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/Flags.show_every, time.time() - time_start))
            train_loss.append(loss_val/Flags.show_every)
            loss_val = 0
            time_start = time.time()
        if batch_id % Flags.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id+1) + ".pt")


        ## VALIDATE ON EVALUATION SET TO CHECK IF MODEL IS LEARNING
        if batch_id % Flags.test_every == 0:
            right, error = 0, 0
            for _, (test1, test2) in enumerate(testLoader, 1):
                if Flags.cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else: error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\taccuracy:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            scheduler.step()



    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    torch.save(net.state_dict(), Flags.model_path + '/FINAL.pt')
