"""Interprete the CNNs using Grad-CAM visualization technique based on the classic book deep learning with python by Francis Chollet and the blog: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82"""

import os
import argparse
import json
from tabulate import tabulate
from sklearn.feature_extraction import image
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19, densenet201, densenet121
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2


NUM_CLASSES=6
NUM_CHANNEL_LAST_CONV_LAYER=1024
SIZE=256
WHICH_STYLE=1 # 0 川派 1 晋派 2 京派 3 闽派 4 苏派 5 皖派

data_root='/media/yinchao/Mastery/dataset/Architectural-Style-Cls/CAM/data'
image_root=f'{data_root}/archiStyle'
# image_raw_name='Jing_2_50.jpg'
image_raw_name='sample.jpg'
# image_cam_name='Jing_2_50_cam_densenet.jpg'
image_cam_name='sample_cam_densenet.jpg'

# TODO: allows to predict images and generate corresponding CAM images
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='experiments-TL',
                    help='Directory containing results of experiments')



def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint


# use hook mechanism to obtain grads
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        
        # get the pretrained DenseNet201 network
        # self.densenet = densenet201(pretrained=True)
        self.densenet_orig = densenet121(pretrained=True)
        self.densenet = densenet121(pretrained=False)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = torch.nn.Sequential(
                        nn.Linear(num_ftrs, 4096),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(4096, 256),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(256, NUM_CLASSES),
                        torch.nn.LogSoftmax(dim=1)) 
        # load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
        load_checkpoint(os.path.join('/media/yinchao/Mastery/dataset/Architectural-Style-Cls/CAM/checkpoints/best'+ '.pth.tar'), self.densenet)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features
        
        # add the average global pool
        self.global_avg_pool = nn.AvgPool2d(kernel_size=8, stride=1) # kernel_size is relied on the input image size
        
        # get the classifier of the vgg19
        self.classifier = self.densenet.classifier

        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x) # (B, 1024, 8, 8)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # don't forget the pooling
        x = self.global_avg_pool(x)
        # x = x.view((1, 1920))
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)

'''
load the given image to the batch form
'''
# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# define a dataset with 1 image 
dataset = datasets.ImageFolder(root=image_root, transform=transform)
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


'''
initialize the VGG model
'''
model = DenseNet()
# set the evaluation mode
model.eval()
# get the image from the dataloader
img, _ = next(iter(dataloader))
# get the most likely prediction of the model
# pred = vgg(img).argmax(dim=1)
pred = model(img)
# the score is log-softmax, so convert back
pred=torch.exp(pred)
# get the gradient of the output with respect to the parameters of the model
pred[:, WHICH_STYLE].backward()
# pull the gradients out of the model
gradients = model.get_activations_gradient()

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

# get the activations of the last convolutional layer
activations = model.get_activations(img).detach()
# weight the channels by corresponding gradients, 512 is the #channels in the conv layer
for i in range(NUM_CHANNEL_LAST_CONV_LAYER):
    activations[:, i, :, :] *= pooled_gradients[i]
# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())


# interpolate the image
# img = cv2.imread('./data/Elephant/data/05fig34.jpg')
img = cv2.imread(f'{image_root}/test/{image_raw_name}')
heatmap = heatmap.numpy()
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
# cv2.imwrite('./map.jpg', superimposed_img)
# cv2.imwrite('/media/yinchao/Mastery/dataset/Architectural-Style-Cls/CAM/data/cam.jpg', superimposed_img)
cv2.imwrite(f'{image_root}/{image_cam_name}', superimposed_img)
