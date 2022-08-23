# encoding: utf8
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .misc import *   
import pdb
import cv2

img_size = 512

__all__ = ['make_image', 'show_batch', 'show_mask', 'show_mask_single', 'visualize_heatmap', 'get_landmarks_from_heatmap']

# functions to show an image
def make_image(img, mean=(0,0,0), std=(1,1,1)):
    for i in range(0, 3):
        img[i] = img[i] * std[i] + mean[i]    # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def show_batch(images, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.imshow(images)
    plt.show()


def show_mask_single(images, mask, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(2, 1, 1)
    plt.imshow(images)
    plt.axis('off')

    # for b in range(mask.size(0)):
    #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
    mask_size = mask.size(2)
    # print('Max %f Min %f' % (mask.max(), mask.min()))
    mask = (upsampling(mask, scale_factor=im_size/mask_size))
    # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
    # for c in range(3):
    #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

    # print(mask.size())
    mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
    # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.axis('off')

def show_mask(images, masklist, Mean=(2, 2, 2), Std=(0.5,0.5,0.5)):
    im_size = images.size(2)

    # save for adding mask
    im_data = images.clone()
    for i in range(0, 3):
        im_data[:,i,:,:] = im_data[:,i,:,:] * Std[i] + Mean[i]    # unnormalize

    images = make_image(torchvision.utils.make_grid(images), Mean, Std)
    plt.subplot(1+len(masklist), 1, 1)
    plt.imshow(images)
    plt.axis('off')

    for i in range(len(masklist)):
        mask = masklist[i].data.cpu()
        # for b in range(mask.size(0)):
        #     mask[b] = (mask[b] - mask[b].min())/(mask[b].max() - mask[b].min())
        mask_size = mask.size(2)
        # print('Max %f Min %f' % (mask.max(), mask.min()))
        mask = (upsampling(mask, scale_factor=im_size/mask_size))
        # mask = colorize(upsampling(mask, scale_factor=im_size/mask_size))
        # for c in range(3):
        #     mask[:,c,:,:] = (mask[:,c,:,:] - Mean[c])/Std[c]

        # print(mask.size())
        mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask.expand_as(im_data)))
        # mask = make_image(torchvision.utils.make_grid(0.3*im_data+0.7*mask), Mean, Std)
        plt.subplot(1+len(masklist), 1, i+2)
        plt.imshow(mask)
        plt.axis('off')


img_size = 512
## get the position of landmarks
x_pos_idxs = np.array(range(img_size)).reshape(1, -1)
x_pos_idxs = np.repeat(x_pos_idxs, img_size, axis=0)
y_pos_idxs = np.array(range(img_size)).reshape(-1, 1)
y_pos_idxs = np.repeat(y_pos_idxs, img_size, axis=1)


def get_landmarks_from_heatmap(pred_heatmap):
    pred_heatmap = np.transpose(pred_heatmap.cpu().numpy(), [1,2,0])
    if pred_heatmap.shape[0] == 512 and pred_heatmap.shape[1] == 512:
        x_idxs = x_pos_idxs
        y_idxs = y_pos_idxs
    else:
        w,h,_ = pred_heatmap.shape
        ## get the position of landmarks
        x_idxs = np.array(range(w)).reshape(1, -1)
        x_idxs = np.repeat(x_idxs, h, axis=0)
        y_idxs = np.array(range(h)).reshape(-1, 1)
        y_idxs = np.repeat(y_idxs, w, axis=1)
    
    landmarks = []
    for i in range(pred_heatmap.shape[-1]):
        x_pos = int(np.sum(pred_heatmap[:,:,i] * x_idxs) / np.sum(pred_heatmap[:,:,i]))
        y_pos = int(np.sum(pred_heatmap[:,:,i] * y_idxs) / np.sum(pred_heatmap[:,:,i]))
        landmarks.append([x_pos, y_pos])
    return landmarks


def visualize_heatmap(input, landmarks):
    # img = np.transpose(input.cpu().numpy())[:,:,0]
    img = input.cpu().numpy()[0,:,:]
    img = 255*(img-np.min(img)) / (np.max(img) - np.min(img))
    img = img.astype(np.uint8)
    img = cv2.merge([img, img, img])
    # draw landmarks on image
    for (x_pos,y_pos) in landmarks:
        cv2.circle(img, (x_pos,y_pos), 2, (0,0,255), -1)
    return img
