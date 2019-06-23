"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--input-list', type=str, default='', help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
opts = parser.parse_args()

device = None
if not opts.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('device: {}'.format(device))

torch.manual_seed(opts.seed)
if device == "cuda":
    torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config, device)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config, device)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint, map_location=lambda storage, loc: storage), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.to(device=device)
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

input_list = []

if opts.input_list != '':
    with open(opts.input_list, 'r') as f:
        for line in f.read().split('\n'):
           input_list.append(os.path.join(opts.input, line))
else:
    input_list.append(opts.input)


with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).to(device=device)) if opts.style != '' else None

    # Start testing
    for i, input in enumerate(input_list):
        image = Variable(transform(Image.open(input).convert('RGB')).unsqueeze(0).to(device=device))
        content, _ = encode(image)

        if opts.trainer == 'MUNIT':
            style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).to(device=device))
            if opts.style != '':
                _, style = style_encode(style_image)
            else:
                style = style_rand

            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                path = os.path.join(opts.output_folder, '{:03d}-output{:03d}.jpg'.format(i, j))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
        elif opts.trainer == 'UNIT':
            outputs = decode(content)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output.jpg')
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        else:
            pass

        if not opts.output_only:
            # also save input images
            vutils.save_image(image.data, os.path.join(opts.output_folder, '{:03d}-input.jpg'.format(i)), padding=0, normalize=True)

