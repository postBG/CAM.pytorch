import os
import argparse

import torch

from models import ConvNetWithGAP

parser = argparse.ArgumentParser(description='PyTorch CAM')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--epoch', type=int, default=100, help='epoch (default: 100)')
parser.add_argument('--output', type=str, default='output', help='dir to save model (default: output)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
args = parser.parse_args()


def main():
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetWithGAP()


if __name__ == "__main__":
    main()
