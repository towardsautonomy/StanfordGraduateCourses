import os  
import argparse as arg  
import time  

import torch   

import numpy as np  
import cv2   
from PIL import Image   
from glob import glob
import matplotlib.pyplot as plt

from problem4.model import DenseDepth
from problem4.utils import colorize, DepthNorm, AverageMeter, load_images
from problem4.losses import ssim as ssim_criterion
from problem4.losses import depth_loss as gradient_criterion

def test(checkpoint, 
        device="cuda", 
        data="examples/"):

    if len(checkpoint) and not os.path.isfile(checkpoint):
        raise FileNotFoundError("{} no such file".format(checkpoint))    

    device = torch.device("cuda" if device == "cuda" else "cpu")
    print("Using device: {}".format(device))

    # Initializing the model and loading the pretrained model 
    ckpt = torch.load(checkpoint)
    model = DenseDepth(encoder_pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print("model load from checkpoint complete ...")

    # Get Test Images  
    img_list = glob(data+"*.png")
    
    # Set model to eval mode 
    model.eval()

    # Begin testing loop 
    print("Begin Test Loop ...")

    fig, axs = plt.subplots(3, 2)
    for idx, img_name in enumerate(img_list):

        img = load_images([img_name])
        if idx < 3:     
            axs[idx, 0].imshow(img[0].transpose(1, 2, 0))
            axs[idx, 0].axis('off')
        img = torch.Tensor(img).float().to(device)   

        with torch.no_grad():
            preds = DepthNorm(model(img).squeeze(0))            

        output = colorize(preds.data)
        output = output.transpose((1, 2, 0))
        if idx < 3:     
            axs[idx, 1].imshow(output, cmap='gray')
            axs[idx, 1].axis('off')
        cv2.imwrite(img_name.split(".")[0].replace(data,data+'output/')+"_result.png", output)

        print("Processing {} done.".format(img_name))
    plt.show()

