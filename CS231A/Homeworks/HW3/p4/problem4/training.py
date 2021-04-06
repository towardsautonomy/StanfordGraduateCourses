import time  
import argparse as arg 
import datetime
import os

import torch  
import torch.nn as nn  
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.utils as vision_utils  
from tensorboardX import SummaryWriter

from problem4.losses import ssim as ssim_criterion
from problem4.losses import depth_loss as gradient_criterion
from problem4.data import get_data_loaders
from problem4.utils import AverageMeter, DepthNorm, colorize, init_or_load_model

def train(epochs, 
        trainloader,
        testloader,
        lr=0.0001, 
        save="checkpoints/", 
        theta=0.1, 
        device="cuda", 
        pretrained=True,
        checkpoint=None):

    num_trainloader = len(trainloader)
    num_testloader = len(testloader)

    # Training utils  
    model_prefix = "monocular_"
    device = torch.device("cuda:0" if device == "cuda" else "cpu")
    theta = theta
    save_count = 0
    epoch_loss = []
    batch_loss = []
    sum_loss = 0

    if checkpoint:
        print("Loading from checkpoint ...")
        
        model, optimizer, start_epoch = init_or_load_model(pretrained=pretrained,
                                                            epochs=epochs,
                                                            lr=lr,
                                                            ckpt=checkpoint, 
                                                            device=device                                            
                                                            )
        print("Resuming from: epoch #{}".format(start_epoch))

    else:
        print("Initializing fresh model ...")

        model, optimizer, start_epoch = init_or_load_model(pretrained=pretrained,
                                                            epochs=epochs,
                                                            lr=lr,
                                                            ckpt=None, 
                                                            device=device                                            
                                                            )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if pretrained:
        log_dir = 'runs/pretrained'
    else:
        log_dir = 'runs/not_pretrained'
    # Logging
    writer = SummaryWriter(log_dir,comment="{}-training".format(model_prefix))
    
    # Loss functions 
    l1_criterion = nn.L1Loss() 

    # Starting training 
    print("Starting training ... ")
     
    for epoch in range(start_epoch, epochs):
        
        model.train()
        model = model.to(device)

        batch_time = AverageMeter() 
        loss_meter = AverageMeter() 

        epoch_start = time.time()
        end = time.time()

        for idx, batch in enumerate(trainloader):

            optimizer.zero_grad() 

            image_x = torch.Tensor(batch["image"]).to(device=device)
            depth_y = torch.Tensor(batch["depth"]).to(device=device)

            normalized_depth_y = DepthNorm(depth_y)

            preds = model(image_x) 

            # calculating the losses 
            l1_loss = l1_criterion(normalized_depth_y, preds) 
            
            ssim_loss = torch.clamp(
                (1-ssim_criterion(preds, normalized_depth_y, 1000.0/10.0))*0.5, 
                min=0, 
                max=1
            )

            gradient_loss = gradient_criterion(normalized_depth_y, preds, device=device)

            net_loss = (1.0 * ssim_loss) + (1.0 * torch.mean(gradient_loss)) + (theta * torch.mean(l1_loss))
           
            loss_meter.update(net_loss.data.item(), image_x.size(0))
            net_loss.backward()
            optimizer.step()

            # Time metrics 
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(num_trainloader-idx))))

            # Logging  
            num_iters = epoch * num_trainloader + idx  
            if idx % 5 == 0 :
                print(
                    "Epoch: #{0} Batch: {1}/{2}\t"
                    "Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t"
                    "eta {eta}\t"
                    "LOSS (current/average) {loss.val:.4f}/{loss.avg:.4f}\t"
                    .format(epoch, idx, num_trainloader, batch_time=batch_time, eta=eta, loss=loss_meter)
                )

                writer.add_scalar("Train/Loss", loss_meter.val, num_iters)
            if idx%1000 == 0:
                if pretrained:
                    ckpt_path = save+"ckpt_{}_pretrained.pth".format(epoch)
                else:
                    ckpt_path = save+"ckpt_{}_not_pretrained.pth".format(epoch)
                torch.save({
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict":  optimizer.state_dict(),
                    "loss": loss_meter.avg
                }, ckpt_path) 

                LogProgress(model, writer, testloader, num_iters, device)
            del image_x
            del depth_y
            del preds          
        
        print(
            "----------------------------------\n"
            "Epoch: #{0}, Avg. Net Loss: {avg_loss:.4f}\n"
            "----------------------------------"
            .format(
                epoch, avg_loss=loss_meter.avg
            )
        )

def LogProgress(model, writer, test_loader, epoch, device):
    
    """ To record intermediate results of training""" 

    model.eval() 
    sequential = test_loader
    sample_batched = next(iter(sequential))
    
    image = torch.Tensor(sample_batched["image"]).to(device)
    depth = torch.Tensor(sample_batched["depth"]).to(device)
    
    if epoch == 0:
        writer.add_image("Train.1.Image", vision_utils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0:
        writer.add_image("Train.2.Image", colorize(vision_utils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    
    output = DepthNorm(model(image))

    writer.add_image("Train.3.Ours", colorize(vision_utils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image("Train.4.Diff", colorize(vision_utils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    
    del image
    del depth
    del output
