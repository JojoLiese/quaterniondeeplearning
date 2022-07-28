import torch
import numpy as np

# returns the number of correct predictions for each element checking the n most most likely predictions
def topncorrect(n, output, y):
    output = output.detach()
    x = [torch.Tensor.tolist(torch.flip(np.argsort(output[ind,:]), [0])[0:n]) for ind in range(len(output[:, 0]))]
    x = torch.FloatTensor(x)
    count = sum([1 for i in range(len(x[:, 0])) if y[i] in x[i, :]])

    return count

# converting normalized images to work with imshow()
def convert_to_imshow_format(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    trimg = np.transpose(npimg, (1, 2, 0))
    return trimg

# put gaussian noise over one image
def noisy(image, variance=0.1):
    row,col,ch= image.shape
    mean = 0
    var = variance
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

# put gaussian noise over stack of images
def noisymult(images, sigma=0.1):
    nr,row,col,ch = images.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(nr,row,col,ch))
    gauss = gauss.reshape(nr,row,col,ch)
    noisy = images + gauss
    return noisy
