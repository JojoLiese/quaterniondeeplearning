from fullyconnectedlayer import createreallayer, connectlayers, createquatlayer, declarefirstlayer, forwardlayerreal, forwardlayerquat
from convolutionallayer import createquatconvlayer, forwardconv, quatmaxpool
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools import topncorrect, convert_to_imshow_format
import torchvision
import torchvision.transforms as transforms
from torch import optim
import torch.nn as nn
from sklearn import metrics
import seaborn as sns
import gc

# proof of concept on static dataset with now train-test-split

# preparation of Cifar10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchsize = 30

trainset = torchvision.datasets.CIFAR10(root='/home/CIFAR-10PyTorch/data/',
                                        train=True,
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batchsize,
                                          shuffle=True)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# select subset of classes to train on for now
classes = ['cat', 'deer', 'ship']

chosenclasses = [3, 4, 8]
mapping = {3: 0, 4: 1, 8: 2}

trainset.data = [trainset.data[ind] for ind, x in enumerate(trainset.targets) if (x == 3) | (x == 4) | (x == 8)]
trainset.targets = [x for ind, x in enumerate(trainset.targets) if (x == 3) | (x == 4) | (x == 8)]
trainset.targets = [mapping.get(number, number) for number in trainset.targets]

dataiter = iter(trainloader)

# creating the architecture of the easy network
conv1 = createquatconvlayer(2, 4)
layer1 = createquatlayer(32)
layer2 = createreallayer(3)
params1 = declarefirstlayer(162, layer1)
params12 = connectlayers(layer1, layer2)

# defining optimizer and loss using torch, passing in learnable parameters
optimizer = optim.SGD([params1, params12, conv1], lr=0.4, momentum=0.2)

# training and testing on the same data every step to prove, that learning an approximation is possible
images, y = dataiter.next()


f = plt.figure()
for i in range(30):
    f.add_subplot(3, 10, i+1)
    plt.imshow(convert_to_imshow_format(images[i,:,:,:]))
    plt.xlabel(classes[y[i]])
    plt.xticks([])
    plt.yticks([])
plt.show()


# collecting training, loss scores for plot
training = []
lossplot = []
topcorrect = []

# preparations for learning loop
loss = nn.CrossEntropyLoss()
l = 1000
m = nn.Softmax(dim=1)
i = 0
activation = torch.nn.ReLU()
batchnumber = 30
currentrate = optimizer.param_groups[0]['lr']
print(f'Learning {batchnumber} batches of size {batchsize} starting with a learning rate of {currentrate}.')

# learning loop
while i < batchnumber:
    i += 1
    optimizer.zero_grad()
    # forward pass
    convout = forwardconv(images, conv1)
    convout = quatmaxpool(convout, 3, 3)
    convout = convout.permute(0, 2, 1, 3, 4)
    flattened = torch.flatten(convout, start_dim=2)

    try:
        out = activation(forwardlayerquat(flattened, params1))
    except:
        print("adjust size of input into first neuronal layer to fit size of flattened in 'declarefirstlayer'")
        print(flattened.size())
        break

    outflat = torch.transpose(out, 2, 1).flatten(start_dim=1)
    out = activation(forwardlayerreal(outflat, params12))
    output = out.float()

    prediction = torch.argmax(m(output), 1)
    likelihood = torch.max(m(out), 1)
    l = loss(output, y.long())
    difference = prediction-y
    correct = torch.numel(difference[difference==0])
    #top2 = topncorrect(2, m(output), y)
    #topcorrect.append(top2 / len(y))

    training.append(correct / len(y))

    # backpropagation and update
    l.backward()
    optimizer.step()
    ll = l.detach()
    lossplot.append(ll)
    currentrate = optimizer.param_groups[0]['lr']
    if (i) % 10 == 0:
        optimizer.param_groups[0]['lr'] = currentrate * optimizer.param_groups[0]['momentum']
    print(
        f'\nbatch {i}:    correct: {correct}/{len(y)},   loss: {l:.4f},   learning rate: {currentrate:.5f}')

    del convout, flattened, outflat
    gc.collect()

# plot training curve
plt.subplot(211)
plt.plot(list(range(1, i + 1)), np.array(training), 'g-')
#plt.plot(list(range(1, i + 1)), np.array(topcorrect), 'g--', label='top 2 training')
#plt.legend(loc="lower right")
plt.xlabel('batch')
plt.ylabel('correct [%]')
plt.grid(axis="y")
plt.subplot(212)

plt.plot(list(range(1, i + 1)), lossplot, 'r-')
plt.grid(axis="y")
plt.xlabel('batch')
plt.ylabel('loss')
plt.tight_layout()
plt.show()

# Display the visualization of the Confusion Matrix
cf_matrix = metrics.confusion_matrix(y, prediction)

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix Cifar10 with Quaternion Neural Network\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(classes)
ax.yaxis.set_ticklabels(classes)
plt.show()

f = plt.figure()
for i in range(30):
    f.add_subplot(3, 10, i+1)
    plt.imshow(convert_to_imshow_format(images[i,:,:,:]))
    plt.xlabel(classes[y[i]] +": "+classes[prediction[i]])
    plt.xticks([])
    plt.yticks([])
plt.show()
