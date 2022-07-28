import torch
import numpy as np
import matplotlib.pyplot as plt
from tools import noisymult, convert_to_imshow_format
import torchvision
import torchvision.transforms as transforms




# preparation of Cifar10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchsize = 15

trainset = torchvision.datasets.CIFAR10(root='/home/CIFAR-10PyTorch/data/',
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batchsize,
                                          shuffle=True)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

dataiter = iter(trainloader)

images, y = dataiter.next()

imageblurred1 = noisymult(images, 0.05)
imageblurred2 = noisymult(images, 0.10)
imageblurred3 = noisymult(images, 0.20)
imageblurred4 = noisymult(images, 0.30)


# printing examples of different noise levels
f = plt.figure()
for i in range(3):
    f.add_subplot(3, 5, 5*i + 1)
    plt.imshow(convert_to_imshow_format(images[i,:,:,:]))
    plt.xticks([])
    plt.yticks([])
    f.add_subplot(3, 5, 5 * i + 2)
    plt.imshow(convert_to_imshow_format(imageblurred1[i, :, :, :]))
    plt.xticks([])
    plt.yticks([])
    f.add_subplot(3, 5, 5 * i + 3)
    plt.imshow(convert_to_imshow_format(imageblurred2[i, :, :, :]))
    plt.xticks([])
    plt.yticks([])
    f.add_subplot(3, 5, 5 * i + 4)
    plt.imshow(convert_to_imshow_format(imageblurred3[i, :, :, :]))
    plt.xticks([])
    plt.yticks([])
    f.add_subplot(3, 5, 5 * i + 5)
    plt.imshow(convert_to_imshow_format(imageblurred4[i, :, :, :]))
    plt.xticks([])
    plt.yticks([])
plt.show()
