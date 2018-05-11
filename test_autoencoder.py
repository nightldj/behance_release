# test autoencoder for images
# Zheng Xu, xuzhustc@gmail.com, Jan 2018


#reference:
# AAE: https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/
# https://github.com/SherlockLiao/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/06_Autoencoder_Model_Save/Autoencoder_Model_Save.py

#usage
#python test_autoencoder.py  --content-data  data/rand10_set/content --style-data data/rand10_set/style --enc-model models/vgg_normalised_conv5_1.t7 --dec-model none  --dropout 0.5 --gpuid 0 --train-dec --dec-last tanh --trans-flag adin  --diag-flag batch --ae-mix mask --ae-dep E5-E4 --base-mode c4 --st-layer 4w --test-dp --save-image output/output_rand10/face_mask --dise-model  models/behance_release.pth
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import torch as th
from torch.autograd import Variable

import torchvision as thv
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as func

import torch.optim as optim
import torch.backends.cudnn as cudnn


import utils
import load_data as ld
import make_opt as mko
from my_autoencoder import *
import folder

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


import argparse
import os
import time
from datetime import datetime
import shutil

args = utils.get_autoencoder_args()
print '%s_%s'%(args.dataset, args.model)
print datetime.now(), args, '\n============================'


use_cuda = th.cuda.is_available() and not args.use_cpu
dtype = th.cuda.FloatTensor if use_cuda else th.FloatTensor

th.manual_seed(args.seed)
gids = args.gpuid.split(',')
gids = [int(x) for x in gids]
print 'deploy on GPUs:', gids
if use_cuda:
    if len(gids) == 1:
        th.cuda.set_device(gids[0])
    else:
        th.cuda.set_device(gids[0])
        print 'use single GPU', gids[0]
    th.cuda.manual_seed(args.seed)


st_cfg = utils.get_dise_cfg(args.st_layers).split(',')
cnt_cfg = utils.get_dise_cfg(args.cnt_layers).split(',')
base_dep = utils.get_base_dep(args.base_mode)
ae = mask_autoencoder(args.ae_flag, args.ae_dep, args.ae_mix, args.dropout, args.train_dec, st_cfg, cnt_cfg, use_sgm=args.dec_last, trans_flag = args.trans_flag,base_dep=base_dep)
ae.load_model(args.enc_model, args.dec_model)
if args.dise_model is not None and args.dise_model.lower() != 'none':
    ae.load_dise_model(args.dise_model)
if use_cuda:
    ae.cuda()
ae.eval()

def open_dp(layer):
    print type(layer)
    if type(layer) == nn.Dropout:
        layer.train()
if args.test_dp:
    ae.apply(open_dp)



if args.diag_flag is not None and args.diag_flag == 'batch': #batch testing of results
    #load data
    transform_test = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
    cnt_set = folder.ImageFolder(args.content_data, transform=transform_test)
    st_set = folder.ImageFolder(args.style_data, transform=transform_test)
    cnt_loader = th.utils.data.DataLoader(cnt_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    st_loader = th.utils.data.DataLoader(st_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    #unload
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    def imsave(tensor, savefile):
        image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
        image = unloader(image)
        image.save(savefile)
        os.chmod(savefile, 0o777)
    #test
    if not os.path.exists(args.save_image):
        os.makedirs(args.save_image)
        os.chmod(args.save_image, 0o777)
    for bi,(inputs,labels) in enumerate(cnt_loader): #iter all content test, random select same number of style images
        l=labels[0]
        if use_cuda:
            inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
        inputs,labels = Variable(inputs,volatile=True), Variable(labels,volatile=True)
        for bj,(st_inputs,st_labels) in enumerate(st_loader): #iter all content test, random select same number of style images
            #if l == st_labels[0]:  #same folder
            #    save_folder = '%s/%s'%(args.save_image, cnt_set.classes[l])
            if True:  #iterate all pairs of content-style folders
                save_folder = '%s/%s_%s'%(args.save_image, cnt_set.classes[l], st_set.classes[st_labels[0]])
                #save_folder = '%s/%s/%s'%(args.save_image, cnt_set.classes[l], st_set.classes[st_labels[0]])
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                    os.chmod(save_folder, 0o777)
                if use_cuda:
                    st_inputs, st_labels = st_inputs.cuda(async=True), st_labels.cuda(async=True)
                st_inputs,st_labels = Variable(st_inputs,volatile=True), Variable(st_labels,volatile=True)

                #forward pass
                img12,_,_,mask = ae(inputs, st_inputs)

                mm = th.mean(mask, dim=1, keepdim=True) #get mean
                mstd = th.mean(mask, dim=1, keepdim=True) #get mean
                mm_img = (th.cat([mm,mm,mm], dim=1) + 1.0) *0.5
                mstd_img = th.cat([mstd,mstd,mstd], dim=1)*0.5

                tmp = inputs.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d.jpg'%(save_folder,bi, bj, 1))
                tmp = st_inputs.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d.jpg'%(save_folder, bi, bj, 2))
                tmp = img12.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d.jpg'%(save_folder, bi, bj, 12))
                tmp = mm_img.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d_maskm.jpg'%(save_folder, bi, bj, 12))
                tmp = mstd_img.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d_maskstd.jpg'%(save_folder, bi, bj, 12))
                mm_img = (mm_img - th.min(mm_img)) /(th.max(mm_img) - th.min(mm_img))
                #print mm.data[0][0], mm_img[0][0]
                tmp = mm_img.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d_maskm2.jpg'%(save_folder, bi, bj, 12))
                mstd_img = (mstd_img - th.min(mstd_img))/(th.max(mstd_img) - th.min(mstd_img))
                tmp = mstd_img.data[0].clamp_(0, 1)
                imsave(tmp, '%s/c%d_s%d_%d_maskstd2.jpg'%(save_folder, bi, bj, 12))
    print 'complete!'

else:  #test for one image
    print 'unsupported testing mode'
