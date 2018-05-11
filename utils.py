#June 9, Zheng Xu, xuzhustc@gmail.com
# some general functions for all training/testing


# -*- coding: utf-8 -*-

import torch as th
from torch.autograd import Variable

import torchvision as thv
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as func

import torch.optim as optim
import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time
from datetime import datetime



def get_autoencoder_args():
    parser = argparse.ArgumentParser(description='PyTorch autoencoder')
    parser.add_argument('--use-cpu', action='store_true', help='use CPU')
    #training
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers to load image')
    parser.add_argument('--lr-freq', default=30, type=int, help='learning rate scheduler, 0.1 every x epochs')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay ')
    parser.add_argument('--epochs', default=80, type=int, help='number of total epochs') #80 for imagenet
    parser.add_argument('--save-model', default='models/', help='folder to save model')
    parser.add_argument('--model', default='vgg19', help='net arch:vgg11,13,16,19')
    parser.add_argument('--dataset', default='disentangle', help='dataset: cifar10;cifar100;imagenet32')
    parser.add_argument('--content-data', default='data/behance_images/crowd_labels_content', help='data path for content images')
    parser.add_argument('--style-data', default='data/behance_images/crowd_labels_media', help='data path for style images')
    parser.add_argument('--save-freq', default=1e5, type=int, help='save every x epochs')
    parser.add_argument('--print-freq', default=1e5, type=int, help='print every x minibatches')
    parser.add_argument('--batch-size', default=16, type=int, help='train minibatch size')
    parser.add_argument('--test-batch-size', default=8, type=int, help='test minibatch size')
    parser.add_argument('--seed', default=2017, type=int, help='random seed')
    parser.add_argument('--test-flag', action='store_true', help='testing')
    parser.add_argument('--load-model', default=None, type=str, help='load model for fine tuning or testing')
    parser.add_argument('--gpuid', default='0,1,2,3', type=str, help='the GPU IDs')
    parser.add_argument('--optm', default='adam', help='discriminator optimizer: padam | sgd | adam | rmsp')
    parser.add_argument('--g-optm', default='adam', help='generator optimizer: sgd | adam | rmsp')
    parser.add_argument('--adam-b1', default=0.9, type=float, help='adam parameter')
    parser.add_argument('--adam-b2', default=0.999, type=float, help='adam parameter')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    #parser.add_argument('--logfile', default=None, type=str, help='the log file for showing loss curve')
    #test
    parser.add_argument('--save-image',default='debug', help='path for saving the styled image')
    parser.add_argument('--style-image', default="data/style_trans_images/style/picasso.jpg", help='style image path')
    parser.add_argument('--content-image', default="data/style_trans_images/content/house.jpg", help='content image path')
    parser.add_argument('--imsize', default=256, type=int, help='image size for input & output')
    parser.add_argument('--ae-flag', default='wct', type=str, help='autoencoder architecture')
    parser.add_argument('--ae-dep', default='E5-E4', type=str, help='autoencoder enc architecture')
    parser.add_argument('--ae-mix', default='mul', type=str, help='mixture flag for disentangled features:mul, add')
    parser.add_argument('--enc-model', default='models/wct/vgg_normalised_conv3_1.t7', type=str, help='encoder saved model')
    parser.add_argument('--dec-model', default='models/wct/feature_invertor_conv3_1.t7', type=str, help='decoder saved model')
    parser.add_argument('--dise-model', default='models/dae_020604_disentangle_2017_wct_sgd0.100_0.0001_dp0.00_10_mb16_8', type=str, help='disentangle layers')
    parser.add_argument('--st-layers', default='4', type=str, help='#channels for layers of style encoding')
    parser.add_argument('--cnt-layers', default='4', type=str, help='#channels for layers of content encoding')
    #GAN, discriminator
    parser.add_argument('--d-lr', default=0.1, type=float, help='discriminator learning rate')
    #parser.add_argument('--rec-w', default=1, type=float, help='weight for recosntruction loss')
    parser.add_argument('--cla-w', default=1, type=float, help='weight for discriminator classification loss')
    parser.add_argument('--gan-w', default=1, type=float, help='weight for disciminator adversarial loss')
    parser.add_argument('--per-w', default=1, type=float, help='weight for perceptron loss')
    parser.add_argument('--gram-w', default=1, type=float, help='weight for gram loss')
    parser.add_argument('--cycle-w', default=1, type=float, help='weight for cycle consistency')
    #parser.add_argument('--w21', default=1, type=float, help='weight for image 21')
    parser.add_argument('--disc-model', default=None, type=str, help='saved discriminator for GAN training')
    parser.add_argument('--save-run',default='debug', help='path for saving the intemediate results')
    parser.add_argument('--diag-flag',default=None, type=str, help='flag for testing mode, only support batch for release')
    parser.add_argument('--gan-ratio', default=0.3, type=float, help='the probability of mixing reconstructin as real image, decreasing with epochs')
    parser.add_argument('--gr-freq', default=1, type=float, help='gan ratio decreasing frequency')
    parser.add_argument('--train-dec', action='store_true', help='train the decoder')
    parser.add_argument('--disc-dep', default=3, type=float, help='depth of discriminator')
    #parser.add_argument('--patch-dep', default=3, type=float, help='depth of discriminator')
    parser.add_argument('--use-mse', action='store_true', help='use mse for perceptrual & gram loss; otherwise use L1')
    parser.add_argument('--use-proj', action='store_true', help='use projection discriminator')
    #parser.add_argument('--use-cdisc', action='store_true', help='content info for discriminator')
    parser.add_argument('--blur-perc', action='store_true', help='average pooling for the perceptual loss')
    parser.add_argument('--dec-last', default='sigmoid', type=str, help='the last layer of decoder: sigmoid | tanh | hard')
    parser.add_argument('--pred-gm', default=1, type=float, help='the parameter for predictive step of GAN training, prdict discriminator, use 1 as in original prediction method')
    parser.add_argument('--trans-flag', default='adin', type=str, help='the transform method: wct | adin')
    parser.add_argument('--base-mode', default='c3', type=str, help='the mode for the base of encoder/decoder: 4;c3;c4')
    parser.add_argument('--test-dp', action='store_true', help='dropout while testing')
    #parse args
    args = parser.parse_args()
    return args



def get_model_parameters(net):
    n = 0.0
    for para in net.parameters():
        n += para.numel()
    print 'network parameters', n/1.0e6, 'M'

def get_dise_cfg(s): #input is the string
    if s == '3':
        print 'three dise layers'
        return '128,128,128'
    elif s == '4':
        print 'four dise layers'
        return '256,D,256,256,U,128'
    elif s == '4w':
        print 'four wide dise layers'
        return '256,D,512,512,U,256'
    elif s == '5':
        print 'four dise layers'
        return '256,D,256,256,256,U,128'
    else:
        print 'custom dise layers, returned'
        return s

def get_base_dep(s):
    if s == 'c3':
        return {0,1,2,3}
    elif s == 'c4':
        return {0,1,2,3,4}
    elif s == '4':
        return {4}
    else:
        print 'unknown base_dep mode: use c3 as default'
        return {0,1,2,3}
