import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from backbones.resnet import iresnet18, iresnet50
from backbones.cnn import cnn
from data.face_dataset import FaceDataset
from metrics.FocalLoss import FocalLoss
from metrics.metrics import ArcFace, LinearMetric


def train(args):
    if args.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    # Load the dataset
    dataset = FaceDataset(data_root=args.dataset_root, description=args.datalist, shuffle=True, train=True)
    num_classes = dataset.get_class_num()
    train_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers, 
                             pin_memory=args.pin_memory,
                             shuffle=True)

    # Load the model
    if args.backbone == 'iresnet50':
        featureExtractor = iresnet50()
    elif args.backbone == 'iresnet18':
        featureExtractor = iresnet18()
    elif args.backbone == 'cnn':
        featureExtractor = cnn()
    else: raise ValueError("Invalid backbone")
    featureExtractor = featureExtractor.to(device)

    # Define the loss function and metric function
    if args.metric == 'arcface':
        metric_fc = ArcFace(512,
                            class_num=num_classes)
    elif args.metric == 'linear':
        metric_fc = nn.Linear(512, num_classes)
    else: raise ValueError("Invalid metric function")
    metric_fc = metric_fc.to(device)

    if args.loss == 'cross_entropy':
        loss_fc = nn.CrossEntropyLoss()
    elif args.loss == 'focal_loss':
        loss_fc = FocalLoss(gamma=args.gamma)
    else: raise ValueError("Invalid loss function")
    
    if isinstance(args.lr, str):
        args.lr = float(args.lr)
    if isinstance(args.weight_decay, str):
        args.weight_decay = float(args.weight_decay)

    # Define the optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': featureExtractor.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': featureExtractor.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.lr, weight_decay=args.weight_decay)
    else: raise ValueError("Invalid optimizer")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)

    # Load the checkpoint if resume training
    if args.resume:
        try:
            checkpoint_dict = torch.load(os.path.join(args.save_dir, args.resume_file))
            featureExtractor.load_state_dict(checkpoint_dict['model_state_dict'])
            metric_fc.load_state_dict(checkpoint_dict['metric_fc_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            print(f"Checkpoint loaded: {args.resume_file}")
            start_epoch = checkpoint_dict['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except:
            start_epoch = 0
            print("Load checkpoint failed")
            print("Starting training from scratch")
    else:
        start_epoch = 0
        print("Starting training from scratch")

    # Training loop
    logging.info(f"Training {args.dataset} dataset with {args.loss}\n"
                 f"Optimizer: {args.optimizer}, Learning rate: {args.lr}, Weight decay: {args.weight_decay}\n"
                 f"Batch size: {args.batch_size}, Network: {args.backbone}\n")
    featureExtractor.train()
    metric_fc.train()
    print(f"training with {args.backbone} backbone")
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        train_bar = tqdm(train_loader)
        for _, (sample, label) in enumerate(train_bar):
            sample = sample.to(device)
            label = label.to(device).long()

            # Forward pass
            feature = featureExtractor(sample)
            logits = metric_fc(feature, label)
            loss = loss_fc(logits, label)

            # Backward pass
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            logging.info(f'Epoch {epoch}/{args.num_epochs}, Loss: {loss.item()}')

            # Update the progress bar
            if args.show_progress:
                train_bar.set_description(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item():.4f}')
        scheduler.step()
       
        # Save checkpoint
        if args.save_all_states:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': featureExtractor.state_dict(),
                'metric_fc_state_dict': metric_fc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, 
                       os.path.join(args.save_dir, 
                       f'checkpoint_{args.metric}_{args.backbone}_{epoch+1}.pth')
            )
                
# Save the model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(featureExtractor.state_dict(), os.path.join(args.save_dir, f'Model_{args.metric}_{args.backbone}_best.pth'))

if __name__ == '__main__':
    time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logging.basicConfig(filename=f"./Log/{time}-train.log", filemode="w",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                        datefmt="%d-%m-%Y %H:%M:%S", level=logging.INFO)

    cfg = yaml.safe_load(open("config.yml"))
    cfg = cfg['train_config']
    args = argparse.Namespace()
    for key in cfg:
        setattr(args, key, cfg[key])

    train(args)