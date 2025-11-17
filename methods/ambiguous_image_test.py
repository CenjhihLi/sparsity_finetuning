import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import timm
from torchvision import datasets
from torch.autograd import Variable
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, Linear
from losses import DistillationLoss
from methods.data_loader import tiny_imagenet, transform, tiny_imagenet_c
from copy import deepcopy
from matplotlib import pyplot as plt

import json
import argparse 
import os 

SMALL_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) #3, B, self.num_heads, N, self.head_dim
    q, k, v = qkv.unbind(0) # B, self.num_heads, N, self.head_dim
    q, k = self.q_norm(q), self.k_norm(k)  # B, self.num_heads, N, self.head_dim

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # (B, self.num_heads, N, self.head_dim) @ (B, self.num_heads, self.head_dim, N) = (B, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v  # (B, self.num_heads, N, N) @ (B, self.num_heads, N, self.head_dim) = (B, self.num_heads, N, self.head_dim)

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

@torch.no_grad()
def val(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
            out = model(data)
            _, pred = torch.max(out.data, 1)
            print(f"The predictions are {pred}, while the labels are {target}")
        return 

path = "./output/images/val_ambiguous_labels"
val_ambiguous = datasets.ImageFolder(path, transform=transform)
val_ambiguous_loader = torch.utils.data.DataLoader(val_ambiguous, batch_size = 64, shuffle=False)
for images, labels in val_ambiguous_loader:
    for i in range(images.shape[0]):
        img = images[i]
        plt.imshow(img.permute(1,2,0).numpy())
        plt.axis("off")
        print(f"Label {labels[i]}")
        plt.show()

checkpoint_deit_b_dist = "{filepath for deit_base_distilled_patch16_224.pth}"
checkpoint_deit_b = "{filepath for deit_base_patch16_224.pth}"

from timm.models import deit_base_distilled_patch16_224
model = deit_base_distilled_patch16_224(pretrained=True)
model.reset_classifier(200)

model_checkpoint = torch.load(checkpoint_deit_b_dist) 
model.load_state_dict(model_checkpoint['model'])   

for m in model.modules():
    if isinstance(m, timm.models.vision_transformer.Attention):
        #m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) 
        # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
        from types import MethodType
        m.forward = MethodType(forward, m)

print("Prediction by deit base distilled: ")
model.to(DEVICE)
val(model, val_ambiguous_loader)

from timm.models import deit_base_patch16_224
model = deit_base_patch16_224(pretrained=True)
model.reset_classifier(200)

model_checkpoint = torch.load(checkpoint_deit_b) 
model.load_state_dict(model_checkpoint['model'])   
for m in model.modules():
    if isinstance(m, timm.models.vision_transformer.Attention):
        #m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) 
        # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
        from types import MethodType
        m.forward = MethodType(forward, m)

print("Prediction by deit base: ")
model.to(DEVICE)
val(model, val_ambiguous_loader)