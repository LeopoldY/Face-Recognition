import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones.resnet import iresnet18, iresnet50
from data.face_dataset import FaceDataset
from utils.acc_eval import compute_accuracy


def eval(args):
    if args.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # Load the dataset
    eval_dataset = FaceDataset(data_root=args.dataset_root, description=args.datalist, train=False)
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=args.batch_size, 
                             num_workers=args.num_workers, 
                             pin_memory=args.pin_memory)

    # Load the model
    if args.backbone == 'iresnet50':
        featureExtractor = iresnet50()
    elif args.backbone == 'iresnet18':
        featureExtractor = iresnet18()
    else: raise ValueError("Invalid backbone")
    model_dict = torch.load(args.model_path)
    featureExtractor.load_state_dict(model_dict['model_state_dict'])
    featureExtractor = featureExtractor.to(device)
    featureExtractor.eval()

    similarities = []
    labels = []

    eval_bar = tqdm(eval_loader)
    for _, (img1, img2, label) in enumerate(eval_bar):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels.extend(label.cpu().numpy())
        with torch.no_grad():
            feat1 = featureExtractor(img1)
            feat2 = featureExtractor(img2)
            sim = nn.functional.cosine_similarity(feat1, feat2).cpu().numpy()
            similarities.extend(sim)

    acc, threshold = compute_accuracy(similarities, eval_dataset.labels)
    print(f"Dataset: {args.dataset} Evalutation accuracy: {acc:.4f} with threshold {threshold:.4f}")

if __name__ == '__main__':
    cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.FullLoader)
    cfg = cfg['eval_config']
    args = argparse.Namespace()
    for key in cfg:
        setattr(args, key, cfg[key])

    eval(args)