# python script for loading behance dataset
# Zheng Xu, xuzhustc@gmail.com, Jan 2018
# reference: https://git.corp.adobe.com/gist/wilber/0806c280f32507bf628461cc1341b67c


# -*- coding: utf-8 -*-

import folder

import sqlite3
from datetime import datetime
import os
import re
import shutil
from numpy import exp

import torchvision as thv
import torchvision.transforms as transforms


def load_dataset(args):

    if args.dataset == 'face' :
        transform_train = transforms.Compose([
            transforms.Scale(224),  # scale imported image
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
            ])
        transform_test = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #normalize,
            ])
        cnt_trainset = folder.ImageFolder('%s/train'%args.content_data, transform=transform_train)
        cnt_testset = folder.ImageFolder('%s/val'%args.content_data, transform=transform_test)
        cnt_nclass = len(os.listdir('%s/train'%args.content_data))
        print 'content data class #', cnt_nclass, 'imgs #:', len(cnt_trainset.imgs), '/', len(cnt_testset.imgs)
        print 'content classes', cnt_trainset.classes, cnt_testset.classes
        st_trainset = folder.ImageFolder('%s/train'%args.style_data, transform=transform_train)
        st_testset = folder.ImageFolder('%s/val'%args.style_data, transform=transform_test)
        st_nclass = len(os.listdir('%s/train'%args.style_data))
        print 'style data class #', st_nclass, 'imgs #:', len(st_trainset.imgs), '/', len(st_testset.imgs)
        print 'style classes', st_trainset.classes, st_testset.classes
    else:
        print 'unknown dataset'

    return cnt_trainset,cnt_testset,cnt_nclass,st_trainset,st_testset,st_nclass


