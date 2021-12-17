import numpy as np
import torch
import torch.utils.tensorboard as tb
from utils import seg_transforms
import argparse
import time
import datetime
from os import path
import sys
from utils.prepper import load_dataset
from models.fcn import FCN
from models.unet import UNet

def create_config(args):
    config = {
        "resume_training" : bool(args.rt),
        "output_model":bool(args.om),
        "chkpt_name": str(args.cpname),
        "save_every": int(args.save_every),
        "model": str(args.model),
        "num_epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "optimizer_type": str(args.optim),
        "scheduler_type": str(args.scheduler),
        "use_gpu": bool(args.use_gpu),
        "num_workers": int(args.num_workers),
        "focal_loss": bool(args.focal_loss),
        "use_rw": bool(args.use_rw),
        "reweight": float(args.reweight),
        "resize": int(args.resize),   
    }
    return config

def display_config(config):
    print("{")
    for k,v in config.items():
        print("\t", k, " : ", v)
    print("}")
    
def pprint_epoch(pass_type,curr_epoch, total_epochs, curr_lr, epoch_error, time_elapsed):
    epoch_log = "[LOG]:{pass_type} EPOCH {curr}/{total_e} completed! LR:{curr_lr} -- Loss: {epoch_loss:.4f}".format(
        pass_type=pass_type,curr=curr_epoch, total_e=total_epochs, curr_lr=curr_lr, epoch_loss=epoch_error)
    print("*"*80)
    print("Time Elapsed: ", time_elapsed)
    print(epoch_log)
    print("*"*80, "\n\n")

def data_augment(resize=256):
    # import torchvision.transforms as transforms
    transformer = seg_transforms.Compose([
        # dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
        seg_transforms.ToTensor(),
        seg_transforms.Rescale((resize,resize))
    ])

    return transformer

def get_dataloader(transformer,num_workers, batch_size):
    # print("blah", transformer)
    train_dataloader = load_dataset(
        img_path="train/",
        label_path="train_masks/",
        num_workers=num_workers,
        batch_size=batch_size,
        transform=transformer
    )
    return train_dataloader

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

def checkpoint_model(config,model=None,optimizer_type=None,optimizer=None,scheduler_type=None,scheduler=None,curr_epoch=0):
    
    name = f"ss_{config['model']}_checkpoint.th"
    
    state_dict = {
        "model" : model.state_dict(),
        "optimizer_type": optimizer_type,
        "optimizer" : optimizer.state_dict(),
        "scheduler_type": scheduler_type,
        "scheduler" : scheduler.state_dict(),
        "epoch" : curr_epoch,
        "lr" : optimizer.param_groups[0]['lr']
    }
    torch.save(state_dict, path.join(path.dirname(path.abspath(__file__)),'checkpoints', name))

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

def convert_model(config):
    model = get_model(config)
    chkpt = torch.load(path.join(path.dirname(path.abspath(__file__)),'checkpoints', f"ss_{config['model']}_checkpoint.th"),map_location='cpu')
    model.load_state_dict(chkpt['model'])
    save_model(model)
    print("[LOG]: Converted Model!")

class FocalLoss(torch.nn.Module):
    def __init__(self,use_rw=False,weights=None):
        super().__init__()
        if(use_rw):
            self.bce_loss = torch.nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=torch.tensor(weights)
            )
        else:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            
    def forward(self,pred,target):
        bce_loss = self.bce_loss(pred,target)
        classes = 1-(2*target)
        p_t = torch.sigmoid(pred*classes)
        floss = (p_t * bce_loss).mean() / p_t.mean()
        return floss

def get_model(config):
    if(config['model'] == 'fcn'):
        return FCN()
    elif(config['model'] == 'unet'):
        return UNet()

def train(args):
    config = create_config(args)
    device = torch.device("cuda:0" if config["use_gpu"] else "cpu")
    
    # Logger Setup
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    model = None
    optimizer = None
    scheduler = None
    num_epochs = 0
    
    if(config['resume_training']):
        print("=== Loading Checkpoint ===")
        model,optimizer,scheduler,num_epochs = load_checkpoint(config,device)
        print(f"[LOG]: Currently {num_epochs} have been completed!")
    else:
        model = get_model(config)
        optimizer = get_optimizer(optim_type=config['optimizer_type'],model=model,lr=config['learning_rate'])
        scheduler = get_scheduler(scheduler_type=config['scheduler_type'], optimizer=optimizer)

        # Send model to GPU if config wanted
        if(config['use_gpu'] and torch.cuda.is_available()):
            print(f"[LOG]: Using GPU: {torch.cuda.get_device_name(0)}")
            model.to(device)
            
    transformer = data_augment(resize=config['resize'])
    
    train_dataloader = get_dataloader(transformer,num_workers=config['num_workers'], batch_size=config['batch_size'])

    loss_fn = None
    if(config['focal_loss'] and config['use_rw']):
        loss_fn = FocalLoss(use_rw=config['use_rw'],weights=config['reweight'])
    elif(config['focal_loss']):
        loss_fn = FocalLoss()
    elif(config['use_rw']):
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config['reweight']))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    
    print("== Training Initiated ==")
    print(model)
    display_config(config)
    print("LOSS FUNCTION: ", loss_fn)
    
    training_gs = 0
    for epoch in range(config['num_epochs']):
        training_epoch_start = time.time()
        
        # Run training loop
        epoch_loss,training_gs = training_epoch(
            device=device,
            config=config,
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer, 
            scheduler=scheduler,
            logger=train_logger,
            global_step=training_gs
        )
        
        train_logger.add_scalar('train/epoch_loss', epoch_loss, epoch)
        
        # Track Training Forward Pass Time
        training_epoch_end = time.time()
        epoch_time_elapsed = datetime.timedelta(
            seconds=(training_epoch_end-training_epoch_start))
        
        if(epoch % config['save_every'] == 0):
            checkpoint_model(
                config,
                model=model, 
                optimizer_type=config['optimizer_type'], 
                optimizer=optimizer, 
                scheduler_type=config['scheduler_type'],
                scheduler=scheduler, 
                curr_epoch=num_epochs
            )
        
        pprint_epoch(
            pass_type='TRAIN',
            curr_epoch=epoch+1, 
            total_epochs=config['num_epochs'], 
            curr_lr=optimizer.param_groups[0]['lr'], 
            epoch_error=epoch_loss, 
            time_elapsed=epoch_time_elapsed
        )
        
        num_epochs += 1
    
    checkpoint_model(
        config,
        model=model, 
        optimizer_type=config['optimizer_type'], 
        optimizer=optimizer, 
        scheduler_type=config['scheduler_type'],
        scheduler=scheduler, 
        curr_epoch=num_epochs
    )

    # if(config['output_model']):
    #     save_model(model)

def training_epoch(device,config,dataloader,model,loss_fn,optimizer,scheduler,logger,global_step):

    # Set Model to training mode
    model.train()
    
    epoch_loss = 0
    for idx,(img,labels) in enumerate(dataloader):
        
        optimizer.zero_grad()

        img = img.to(device)
        gt_mask = labels.to(device)
        
        logits = model(img)

        loss = loss_fn(logits,gt_mask)
        epoch_loss += loss.item()
        
        logger.add_scalar('train/loss', loss.item(), global_step=global_step)
        seg_log(logger, img, gt_mask, logits, global_step)
        
        loss.backward()
        optimizer.step()
        global_step += 1
        
    return epoch_loss,global_step

def seg_log(logger, imgs, gt_label, pred, global_step,num_examples=4):
    logger.add_images('image', imgs[:num_examples], global_step)
    logger.add_images('label', gt_label[:num_examples], global_step)
    logger.add_images('pred', torch.sigmoid(pred[:num_examples]), global_step)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    
    # Model parameters arguments
    parser.add_argument('-gpu', '--use_gpu', dest='use_gpu',action='store_true', help='Use GPU for training')
    parser.add_argument('-logdir','--log_dir', dest='log_dir')
    parser.add_argument('-convert','--convert-checkpoint', dest='convert',action='store_true')
    parser.add_argument('-rt','--resume-training', dest='rt', action='store_true')
    parser.add_argument('-o','--output-model', dest='om', action='store_true')
    parser.add_argument('-se','--save-every', dest='save_every', type=int, default=1)
    parser.add_argument('-cpname','--chkpt-name',type=str,dest='cpname')
    parser.add_argument('-m','--model',type=str,dest='model')
    parser.add_argument('-e', '--epochs', dest="epochs",type=int, default="10")
    parser.add_argument('-b', '--batch-size',dest="batch_size", type=int, default='8')
    parser.add_argument('-lr', '--learning-rate', dest="lr",type=float, default=1e-3)
    parser.add_argument('-optim', '--optimizer', type=str, dest='optim')
    parser.add_argument('-scheduler', '--scheduler',type=str, dest='scheduler')
    parser.add_argument('-w','--workers', dest='num_workers',type=int,default=0)
    parser.add_argument('-fl','--focal-loss', dest='focal_loss',action='store_true')
    parser.add_argument('-urw','--use-rw', dest='use_rw',action='store_true')
    parser.add_argument('-rw','--reweight', dest='reweight',type=float,default=880.2312)
    parser.add_argument('-resize','--resize', dest='resize',type=int,default=256)
    args = parser.parse_args()
    
    if(args.convert):
        config = create_config(args)
        convert_model(config)
    else:
        train(args)