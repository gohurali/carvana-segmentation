"""
Authors: Gohur Ali, ...
Version: 20211120
"""
import numpy as np
import torch
import torch.utils.tensorboard as tb
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from utils import seg_transforms
import argparse
import time
import datetime
import os
from os import path
import sys
from utils.prepper import load_dataset
from models.fcn import FCN
from models.unet import UNet


def create_config(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "use_gpu": True,
        "inline": bool(args.inline),
        "device": device,
        "model": str(args.model),
        "resize": int(args.resize),
        "use_single_example": bool(args.use_single_example),
        "single_example_path": str(args.single_example_path),
        "use_dir": bool(args.use_dir),
        "dir_path": str(args.dir_path),
        "learning_rate": 1e-3
    }
    return config

def display_config(config):
    print("{")
    for k,v in config.items():
        print("\t", k, " : ", v)
    print("}")

def get_model(config):
    if(config['model'] == 'fcn'):
        return FCN()
    elif(config['model'] == 'unet'):
        return UNet()

def get_optimizer(optim_type, model, lr):
    optimizer = None
    if(optim_type == "SGD"):
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=0,
            weight_decay=1e-4)
    elif(optim_type == "Adam"):
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)
    elif(optim_type == "RMS"):
        optimizer = torch.optim.RMSprop(
            params=model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
    return optimizer

def get_scheduler(scheduler_type, optimizer):
    scheduler = None
    if(scheduler_type == 'Reduce'):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=0.1,
            patience=2,
            verbose=True
        )
    elif(scheduler_type == 'Step'):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 10)
    elif(scheduler_type == 'Multi'):
        milestones = [8]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)
    return scheduler

def load_checkpoint(config,device):
    
    # Load the model based on what is being trained
    model = get_model(config)
    chkpt = torch.load(path.join(path.dirname(path.abspath(__file__)),"checkpoints", f"ss_{config['model']}_checkpoint.th"))
    model.load_state_dict(chkpt['model'])

    if(config['use_gpu'] and torch.cuda.is_available()):
        print(f"[LOG]: Using GPU: {torch.cuda.get_device_name(0)}")
        model.to(device)

    optimizer_type = chkpt['optimizer_type']
    optimizer = get_optimizer(optim_type=optimizer_type, model=model, lr=config['learning_rate'])
    optimizer.load_state_dict(chkpt['optimizer'])
    optimizer.param_groups[0]['lr'] = config['learning_rate']
    scheduler_type = chkpt['scheduler_type']
    scheduler = get_scheduler(scheduler_type=scheduler_type, optimizer=optimizer)
    scheduler.load_state_dict(chkpt['scheduler'])
    curr_epoch = chkpt['epoch']
    return model, optimizer, scheduler, curr_epoch

def inference(args):
    config = create_config(args)
    display_config(config)
    
    # Load the model checkpoint
    model,_,_,curr_epochs = load_checkpoint(config,config['device'])
    model.eval()
    
    if(config['use_single_example']):
        infer_single(config,model,config['inline'])
    elif(config['use_dir']):
        infer_dir(config, model,config['inline'])    
    

def open_img(path):
    img = np.array(Image.open(path).convert('RGB'))
    img = torch.from_numpy(img).permute(2,0,1).float()
    return img

def infer_single(config,model,inline=False):
    # Load the example
    img = open_img(config['single_example_path'])
    img = TF.resize(img, (config['resize'], config['resize'])).to(config['device'])[None]
    
    input_im = img.detach().cpu().squeeze(0).permute(1,2,0).float().numpy()
    
    print("[LOG]: Input shape: ",img.shape)
    
    pred_mask = model(img).squeeze(0).squeeze(0)
    pred_mask = torch.sigmoid(pred_mask).detach().cpu().numpy()
    # mask = np.where(pred_mask > 0.5,255,0).astype(int)
    
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(30,30))
    plt.subplots_adjust(
        left  = 0.0,  # the left side of the subplots of the figure
        right = 0.2,    # the right side of the subplots of the figure
        bottom = 0.1,   # the bottom of the subplots of the figure
        top = 0.9,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.4   # the amount of height reserved for white space between subplot
    )
    
    axes[0].set_title("Input Image")
    axes[0].imshow((input_im / 255).astype(np.float32))
    axes[1].set_title("Predicted Mask")
    axes[1].imshow((pred_mask * 255).astype(np.uint8))
    
    # Save the mask
    plt.imshow(pred_mask)
    
    if(not inline):
        print("[LOG]: Saved image to: ",config['single_example_path'])
        plt.imsave("outputs/mask.png", pred_mask)
    print("[DONE]!")

def infer_dir(config,model,inline_viz=False):
    
    rows = len(os.listdir(config['dir_path']))
    fig, axes = plt.subplots(nrows=rows, ncols=2,figsize=(30,30))
    plt.subplots_adjust(
        left  = 0.0,  # the left side of the subplots of the figure
        right = 0.2,    # the right side of the subplots of the figure
        bottom = 0.1,   # the bottom of the subplots of the figure
        top = 0.9,      # the top of the subplots of the figure
        wspace = 0.0,   # the amount of width reserved for blank space between subplots
        hspace = 0.4   # the amount of height reserved for white space between subplot
    )
    for i,f in enumerate(os.listdir(config['dir_path'])):
        img = open_img(path.join(config['dir_path'],f))
        img = TF.resize(img, (config['resize'], config['resize'])).to(config['device'])[None]

        pred_mask = model(img).squeeze(0).squeeze(0)
        pred_mask = torch.sigmoid(pred_mask).detach().cpu().numpy()

        input_im = img.detach().cpu().squeeze(0).permute(1,2,0).float().numpy()

        axes[i][0].set_title("Input Image")
        axes[i][0].imshow((input_im / 255).astype(np.float32))
        axes[i][1].set_title("Predicted Mask")
        axes[i][1].imshow((pred_mask * 255).astype(np.uint8))
    if(not inline_viz):
        plt.savefig("outputs/masks.png")
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters arguments
    parser.add_argument('-m','--model',type=str,dest='model')
    parser.add_argument('-resize','--resize', dest='resize',type=int,default=256)
    
    # single example params
    parser.add_argument('--use-single-example', dest='use_single_example', action='store_true')
    parser.add_argument('--single-example-path', dest='single_example_path', type=str)
    
    # directory params
    parser.add_argument('--use-dir', dest='use_dir', action='store_true')
    parser.add_argument('--dir-path', dest='dir_path', type=str)
    
    # General params
    parser.add_argument('--inline', dest='inline', action='store_true')
    
    args = parser.parse_args()
    
    inference(args)