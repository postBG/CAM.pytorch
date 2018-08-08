import os
import argparse

import torch

from models import ConvNetWithGAP
from trainer import Trainer
from dataset import get_cifar10_loader

parser = argparse.ArgumentParser(description='PyTorch CAM')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--epoch', type=int, default=100, help='epoch (default: 100)')
parser.add_argument('--output', type=str, default='output', help='dir to save model (default: output)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--data_root', type=str, default='data', help='data root (default: data).')
parser.add_argument('--mode', type=str, default='train', help='mode (default: train, one of [train, cam])')
args = parser.parse_args()


def run_train_mode():
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetWithGAP()

    trainer = Trainer(model, device, get_cifar10_loader,
                      epoch=args.epoch, lr=args.lr, batch_size=args.batch_size, data_root=args.data_root)
    trainer.train()


def run_cam_mode():
    pass


def main():
    if args.mode == 'train':
        run_train_mode()
    else:
        run_cam_mode()


if __name__ == "__main__":
    main()
