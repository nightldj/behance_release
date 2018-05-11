# my autoencoder for images
# Zheng Xu, xuzhustc@gmail.com, Jan 2018


#reference:
# WCT AE: https://github.com/sunshineatnoon/PytorchWCT/blob/master/modelsNIPS.py
# VGG: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# WCT torch/TF: https://github.com/Yijunmaverick/UniversalStyleTransfer, https://github.com/eridgd/WCT-TF



# -*- coding: utf-8 -*-

import torch as th
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as func

import torch.backends.cudnn as cudnn
from torch.utils.serialization import load_lua

import numpy as np

import os
import time
from datetime import datetime
import shutil

from net_utils import *



class autoencoder(nn.Module):
    def __init__(self, flag = 'wct', depth_flag = 'E5-E4', train_dec=True, base_dep = {0 , 1, 2, 3}, gram_dep = {0, 1, 2, 3, 4, 5}, perc_dep = 4, use_sgm=None):  #base_dep has to be smaller than dec dep
        super(autoencoder, self).__init__()
        if flag == 'wct':
            self.flag = flag
            parts = depth_flag.split('-')
            if len(parts) == 1:
                self.dep = int(parts[0][1])
            elif len(parts) == 2:
                self.aux_dep = int(parts[0][1]) #encoder depth for preceptron loss
                self.dep = int(parts[1][1]) #decoder depth
            self.encs = [make_wct_enc_layers(cfg[1])]  #conv1
            for i in range(2, self.aux_dep+1):
                self.encs.append(make_wct_aux_enc_layers(cfg[i-1], cfg[i])) #conv2~5
            self.encs = nn.ModuleList(self.encs)  #compatible with DataParallel
            print 'encoder stacks', len(self.encs), 'of', self.aux_dep
            self.train_dec = train_dec
            self.base_dep = base_dep
            self.gram_dep = gram_dep
            self.perc_dep = perc_dep
            if train_dec:
                self.dec = make_tr_dec_layers(dec_cfg[self.dep], use_sgm=use_sgm)
                print 'autoencoder init: need decoder training'
            else:
                self.dec = make_wct_dec_layers(dec_cfg[self.dep])
                print 'autoencoder init: wct: please load decoder'
        else:
            print 'autoencoder: init: flag not supported: ', flag


    def load_from_torch(self, ptm, thm, th_cfg):
        print ptm, thm
        i = 0
        for layer in list(ptm):
            if isinstance(layer, nn.Conv2d):
                print i, '/', len(th_cfg), ':', th_cfg[i]
                layer.weight = th.nn.Parameter(thm.get(th_cfg[i]).weight.float())
                layer.bias = th.nn.Parameter(thm.get(th_cfg[i]).bias.float())
                i += 1
        print 'wct load torch #convs', len(th_cfg), i


    def load_aux_from_torch(self, ptm, thm, th_cfg, aux_cfg):
        #print ptm, thm
        assert(len(th_cfg) < len(aux_cfg))
        i = 0
        while i < len(th_cfg):
            assert(th_cfg[i] == aux_cfg[i])
            i += 1

        for layer in list(ptm):
            if isinstance(layer, nn.Conv2d):
                print i, '/', len(aux_cfg), ':', aux_cfg[i]
                layer.weight = th.nn.Parameter(thm.get(aux_cfg[i]).weight.float())
                layer.bias = th.nn.Parameter(thm.get(aux_cfg[i]).bias.float())
                i += 1
        print 'wct load aux torch #convs', len(th_cfg), '-', len(aux_cfg), i


    def load_model(self, enc_model = 'models/wct/vgg_normalised_conv5_1.t7', dec_model = None):
        if self.flag == 'wct':
            print 'load encoder from:', enc_model
            vgg = load_lua(enc_model)
            self.load_from_torch(self.encs[0], vgg, th_cfg[1]) #conv1
            for i in range(2, self.aux_dep+1):
                self.load_aux_from_torch(self.encs[i-1], vgg, th_cfg[i-1], th_cfg[i])
            if not self.train_dec and dec_model is not None and dec_model.lower() != 'none':
                print 'load decoder from:', dec_model
                vgg = load_lua(dec_model)
                self.load_from_torch(self.dec, vgg, th_dec_cfg[self.dep])
        else:
            print 'autoencoder: load: flag not supported', flag


    def enc_dec(self, input):
        code = input
        for i in range(len(self.encs)):
            code = self.encs[i](input)
        out = self.dec(code)
        return out





class mask_autoencoder(autoencoder):
    def __init__(self, flag = 'wct', depth_flag = 'E5-E3', mix_flag = 'mask', dropout = 0.5, train_dec=True, st_cfg=[128], cnt_cfg=[128], use_sgm=None, trans_flag='adin', base_dep={4}, blur_flag=False):
        super(mask_autoencoder, self).__init__(flag, depth_flag, train_dec, use_sgm=use_sgm, base_dep=base_dep)
        self.mix_flag = mix_flag
        self.trans_flag = trans_flag
        self.blur_flag = blur_flag
        self.dp = dropout  #dropout for texture part

        self.in_channels = 0  #base is the concatenation of base_dep
        if 0 in self.base_dep:
            self.in_channels += 3
        for i in range(1, self.aux_dep):
            if i in self.base_dep:
                self.in_channels += cfg[i][-1]

        self.senc = make_dise_layers(self.in_channels*2, self.in_channels, st_cfg, use_bn='in', dropout=self.dp, use_sgm='tanh')
        self.dec = make_tr_dec_layers(dec_cfg[self.dep], in_channels=self.in_channels, use_sgm=use_sgm)


    def freeze_base(self):
        for enc in self.encs:
            for param in enc.parameters():
                param.requires_grad = False
        if not self.train_dec:
            for param in self.dec.parameters():
                param.requires_grad = False


    def get_base_perc_gram(self, img, gram_flag=True, blur_flag=False):
        code = img
        bases = []
        grams = []
        if 0 in self.base_dep:
            bases.append(img)
        if gram_flag and self.trans_flag == 'adin' and 0 in self.gram_dep:
            grams.append(get_gram(img))
        for i in range(len(self.encs)):
            code = self.encs[i](code)
            if (i+1) in self.base_dep:
                if i > 0 or 0 not in self.base_dep:
                    bases = [func.avg_pool2d(b, kernel_size=2, stride=2) for b in bases] #downsample
                bases.append(code)
            if gram_flag and self.trans_flag == 'adin' and (i+1) in self.gram_dep:
                grams.append(get_gram(code))
            if (i+1) == self.perc_dep:
                out = code
        if blur_flag:
            out = func.avg_pool2d(out, kernel_size=2, stride=2)  #pooling to make perceptron loss weaker
        base = th.cat(bases, dim=1)
        return base,out,grams


    def forward(self, img1, img2):
        #basic encoding
        base1, perc1, _ = self.get_base_perc_gram(img1,  gram_flag=False, blur_flag = self.blur_flag)
        base2, _, gram2 = self.get_base_perc_gram(img2,  gram_flag=True, blur_flag = self.blur_flag)

        base1 = base1.detach()
        base2 = base2.detach()

        #adain steps
        wct12 = adin_transform2(base1, base2)
        mask = self.senc(th.cat([base1, base2], dim=1))

        #mix and decode
        if self.mix_flag == 'skip':
            code = wct12
        elif self.mix_flag == 'mask':
            code = mask*base1 + (1-mask)*wct12
        img12 = self.dec(code) #mixture, content1, style2
        return img12, perc1, gram2, mask

    def load_dise_model(self, load_model):
        checkpoint = th.load(load_model)
        self.senc.load_state_dict(checkpoint['st_enc'])
        self.dec.load_state_dict(checkpoint['dec'])
        print 'wct_autoencoder: aux st enc layer loaded from:', load_model



