import torchvision.datasets
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.经典的parse

parser = argparse.ArgumentParser(description="glow")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--iter", type=int, default=20000)
parser.add_argument("--n_flow", default=32, type=int, help=" number of flows blocks")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")


# parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

# 2.经典第二部，构建sample
def sample_data(batch_size, image_size, valid):
    """
    经典的使用自带数据集，然后transform，构建datasets
    之后使用打他loader进行读取
    :param batch_size:
    :param image_size:
    :param valid:
    :return:
    """
    if valid:
        dataset = torchvision.datasets.CelebA(
            "celebA_data",
            split='test',
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Resize(image_size),
                 torchvision.transforms.CenterCrop(image_size),
                 torchvision.transforms.ToTensor()]
            )
        )
        # dataloader 进行读取datasets，并行，还有batch size
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            drop_last=True, num_workers=8
        )
        loader = iter(DataLoader)

        while True:
            try:
                yield next(loader)
            except StopIteration:
                yield None
    else:
        # 构建train训练集
        dataset = torchvision.datasets.CelebA(
            "celebA_data",
            split='train',
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.Resize(image_size),
                 torchvision.transforms.CenterCrop(image_size),
                 torchvision.transforms.ToTensor()]
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            drop_last=True, num_workers=8
        )
        loader = iter(DataLoader)

        while True:
            try:
                yield next(loader)
            except StopIteration:
                loader = DataLoader(
                    dataset, shuffle=True, batch_size=batch_size,
                    num_workers=8
                )
                loader = iter(loader)
                yield next(loader)


# 经典的第三步，进行train
def calc_z_shapes(n_channel, img_size, n_flow, n_block):
    z_shapes = []
    for i in range(n_block - 1):
        img_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, img_size, img_size))

    img_size //= 2
    z_shapes.append((n_channel * 4, img_size, img_size))
    return z_shapes


def calc_loss(log_p, logdet, image_size,n_channel, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * n_channel

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet.mean() + log_p.mean()

    return (
        (-loss / (log(2) * n_pixel)),
        (log_p.mean() / (log(2) * n_pixel)),
        (logdet.mean() / (log(2) * n_pixel)),
    )


def train(args, model, optimizer):
    dataset = iter(sample_data(args.batch, args.img_size, valid=False))
    # 这是已经dataloader读取后的，使用next读取
    n_bins = 2.0 ** args.n_bits
    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        # 产生随机变量z空间，大小如下
        z_new = torch.randn(args.n_sample, *z)
        z_sample.append(z_new)

    classifier_loss_fn = torch.nn.CrossEntropyLoss()
    # 使用tqdm进行迭代
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, label = next(dataset)
            label = torch.argmax(label, dim=1)
            image = image * 255
            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))
            image = image / n_bins - 0.5
            image = image.to(device)
            label = label.to(device)
            if i==0:
                with torch.no_grad():
                    log_p,logdet,z,logits=model(
                        image+torch.rand_like(image,device=device)/n_bins,label
                    )
                    continue
            else:
                log_p, logdet, z, logits=model(image+torch.rand_like(image,device=device)/n_bins,label)

            classifier_loss =classifier_loss_fn(logits,label)

            logdet =logdet.mean()
            nll_loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size,args.n_channel, n_bins)
            loss =nll_loss+classifier_loss
            model.zero_grad()
            loss.backward()
            warmup_lr=args.lr
            optimizer.param_group[0]["lr"]=warmup_lr

            optimizer.step()
            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )