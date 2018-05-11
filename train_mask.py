# gan to improve style transfer
# Zheng Xu, xuzhustc@gmail.com, Jan 2018
#

#reference:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


#usage:
#training
#python train_mask.py  --dataset face --content-data  data/behance_images/faces_align_content_gender --style-data  data/behance_images/faces_behance_lfw_celebA_dtd --enc-model models/vgg_normalised_conv5_1.t7 --dec-model none  --epochs 150 --lr-freq 60 --batch-size 56 --test-batch-size 24 --num-workers 8  --print-freq 200 --dropout 0.5 --g-optm adam  --lr 0.0002 --optm padam --d-lr 0.00002 --adam-b1 0.5  --weight-decay 0 --ae-mix mask --dise-model none --cla-w 1 --gan-w 1 --per-w 1 --gram-w 200 --cycle-w 0 --save-run debug_gan --gpuid 0,1,2,3 --train-dec --use-proj --dec-last tanh --trans-flag adin --ae-dep E5-E4 --base-mode c4  --st-layer 4w --seed 2017

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
from make_loss import *
from my_discriminator import *

import matplotlib.pyplot as plt
import numpy as np

import argparse
import os
import time
from datetime import datetime
import shutil
import math
import random

args = utils.get_autoencoder_args()
print '%s_%s'%(args.dataset, args.ae_flag)
print datetime.now(), args, '\n============================'

tag='gan-mask_%s_%s_%d_%s_%s_%s_%s_c%.3f_a%.3f_p%.3f_g%.3f_cw%.3f_%s%.4f_%s%.4f_p%.1f_wd%.1f_dp%.2f_ep%d_%d_mb%d_%d_gr%.1f_%d_st%s_cnt%s_%s_%s'%(
        datetime.now().strftime("%m%d%H"), args.dataset, args.seed, args.ae_flag, args.ae_dep, args.ae_mix, args.trans_flag,
        args.cla_w, args.gan_w, args.per_w, args.gram_w, args.cycle_w,
        args.g_optm, args.lr, args.optm, args.d_lr, args.pred_gm, args.weight_decay, args.dropout,  args.epochs, args.lr_freq, args.batch_size, args.test_batch_size, args.gan_ratio, args.gr_freq,
        args.st_layers, args.cnt_layers, args.dec_last, args.base_mode)


def get_save_file(args):
    best_file1 = '%s/%s'%(args.save_model, tag)
    return best_file1

def get_debug_folder(args):
    best_file1 = '%s/%s'%(args.save_run, tag)
    return best_file1

use_cuda = not args.use_cpu
random.seed(args.seed)
th.manual_seed(args.seed)
gids = args.gpuid.split(',')
gids = [int(x) for x in gids]
print 'deploy on GPUs:', gids
if use_cuda:
    if len(gids) == 1:
        th.cuda.set_device(gids[0])
    th.cuda.manual_seed(args.seed)

if not os.path.exists(args.save_model):
    os.makedirs(args.save_model)
    os.chmod(args.save_model, 0o777)


# ============================ load data
cnt_trainset,cnt_testset,cnt_nclass,st_trainset,st_testset,st_nclass = ld.load_dataset(args)


cnt_trainloader = th.utils.data.DataLoader(cnt_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
cnt_testloader = th.utils.data.DataLoader(cnt_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
st_trainloader = th.utils.data.DataLoader(st_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
st_testloader = th.utils.data.DataLoader(st_testset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)  #shuffle style

print datetime.now(), 'data loaded!\n'

# ========================== net define and load
def freeze_generator(tmp):
    for p in tmp.senc.parameters():
        p.requires_grad = False  #not update generator
    if args.train_dec:
        for p in tmp.dec.parameters():
            p.requires_grad = False  #not update generator


def unfreeze_generator(tmp):
    for p in tmp.senc.parameters():
        p.requires_grad = True  #not update generator
    if args.train_dec:
        for p in tmp.dec.parameters():
            p.requires_grad = True  #not update generator


class ae_gan(nn.Module):  #combine generator, discriminator and loss to parallel

    def __init__(self, args):
        super(ae_gan, self).__init__()

        st_cfg = utils.get_dise_cfg(args.st_layers).split(',')
        cnt_cfg = utils.get_dise_cfg(args.cnt_layers).split(',')
        print 'style layers', st_cfg, 'content layers', cnt_cfg
        base_dep = utils.get_base_dep(args.base_mode)
        net = mask_autoencoder(args.ae_flag, args.ae_dep, args.ae_mix, args.dropout, args.train_dec, st_cfg, cnt_cfg, use_sgm=args.dec_last, trans_flag = args.trans_flag, base_dep=base_dep, blur_flag = args.blur_perc)
        net.load_model(args.enc_model, args.dec_model)

        #load pre-trained mask layers
        if args.dise_model is not None and args.dise_model != 'none' and args.dise_model != 'None':
            net.load_dise_model(args.dise_model)

        #load pre-trained discriminator layers
        dnet = PatchDiscriminator(scn=st_nclass, ccn=cnt_nclass, n_layers=args.disc_dep, use_proj=args.use_proj, use_cnt=False)
        if args.disc_model is not None and args.disc_model != 'none' and args.disc_model != 'None':
            dnet.load_model(args.dise_model)

        print '================== net \n', net
        utils.get_model_parameters(net)
        print '================== dis net \n', dnet
        utils.get_model_parameters(dnet)

        #loss and optim
        criterion = gan_style_loss()

        net.freeze_base()

        self.net = net
        self.dnet = dnet
        self.criterion = criterion

    def train_disc(self, inputs, labels, st_inputs, st_labels, args, running_dcl, running_dbl, use_real):
        net = self.net
        dnet = self.dnet
        criterion = self.criterion

        for p in dnet.parameters():
            p.requires_grad = True   #update discriminator
        freeze_generator(net)

        #autoencoder forward
        img12,_,_,_ = net(inputs, st_inputs)

        #discriminator for classification
        bc_out12,cc_out12,sc_out12 = dnet(img12.detach(), slabel=st_labels, clabel=labels)
        bc_out2,_,sc_out2 = dnet(st_inputs, slabel=st_labels)

        #loss
        criterion.reset()

        if args.cla_w > 0:
            criterion.add_ce(sc_out12, st_labels, args.cla_w) #style of syth
            criterion.add_ce(sc_out2, st_labels, args.cla_w) #style of input real
            running_dcl += sum([x.data[0] for x in criterion.ls[:2]])

        if args.gan_w > 0:
            if use_real:
                criterion.add_gan_fake(bc_out12, args.gan_w) #gan, fake
                criterion.add_gan_real(bc_out2, args.gan_w) #gan, real
            else:                    #flip label for stable training
                criterion.add_gan_real(bc_out12, args.gan_w)
                criterion.add_gan_fake(bc_out2, args.gan_w)
            running_dbl += sum([x.data[0] for x in criterion.ls[-2:]])

        #backward and update
        if criterion.pls is not None and len(criterion.pls) > 0 :
            print 'warning: parallel loss for discriminator not considered'
        if criterion.ls is not None and len(criterion.ls) > 0 :
            loss = criterion.return_loss()
        else:
            loss = []
        return  loss, running_dcl, running_dbl

    def forward(self, inputs, labels, st_inputs, st_labels, args):
        net = self.net
        dnet = self.dnet
        criterion = self.criterion

        #autoencoder forward, reconstruct and exchange, probably can be comment to save one forward pass
        img12,ftr1,grm2,aux12 = net(inputs, st_inputs)

        #discriminator for classification
        bc_out12,cc_out12,sc_out12 = dnet(img12, slabel=st_labels, clabel=labels)

        #loss
        criterion.reset()

        if args.cla_w > 0:
            criterion.add_ce(sc_out12, st_labels, args.cla_w) #style of syth

        if args.gan_w > 0:
            criterion.add_gan_real(bc_out12, args.gan_w) #gan, fool discriminator

        #perceptron & gram loss
        if args.per_w > 0 or args.gram_w > 0 or args.cycle_w > 0:
            base12,ftr12,grm12 = net.get_base_perc_gram(img12, blur_flag=args.blur_perc)
            assert(len(grm12) == len(grm2))
            if args.cycle_w > 0:
                cycle12 = net.senc(th.cat([base12.detach(), base12.detach()], dim=1))
            if args.use_mse:
                criterion.add_mse(ftr12, ftr1.detach(), args.per_w)
                for i in range(len(grm12)):
                    criterion.add_mse(grm12[i], grm2[i].detach(), args.gram_w)
                if args.cycle_w > 0:
                    criterion.add_mse(aux12, cycle12.detach(), args.cycle_w)
            else:
                criterion.add_l1(ftr12, ftr1.detach(), args.per_w)
                for i in range(len(grm12)):
                    criterion.add_l1(grm12[i], grm2[i].detach(), args.gram_w)
                if args.cycle_w > 0:
                    criterion.add_l1(aux12, cycle12.detach(), args.cycle_w)

        #backward and update
        if criterion.pls is not None and len(criterion.pls) > 0 :  #parallel loss
            print 'warning: parallel loss for generator not considered'
        if criterion.ls is not None and len(criterion.ls) > 0 :  #single gpu loss
            loss = criterion.ls
        else:
            loss = []

        return loss,img12


wrap_net = ae_gan(args)
if args.train_dec:
    optimizer = mko.get_optimizer_var([{'params':wrap_net.net.senc.parameters(), 'params':wrap_net.net.dec.parameters()}], args, args.g_optm, args.lr)
else:
    optimizer = mko.get_optimizer_var([{ 'params':wrap_net.senc.parameters()}], args, args.g_optm, args.lr)
doptimizer = mko.get_optimizer_var(wrap_net.dnet.parameters(), args, args.optm, args.d_lr)

print optimizer,doptimizer

def save_model(epoch, wrap_net,  optimizer, doptimizer, save_file):
    if len(gids) > 1:
        th.save({'epoch':epoch,
            'st_enc':wrap_net.module.net.senc.state_dict(),
            'dec':wrap_net.module.net.dec.state_dict(),
            'disc':wrap_net.module.dnet.state_dict(),
            'optimizer':optimizer.state_dict(),
            'doptimizer':doptimizer.state_dict(),
            }, save_file)
    else:
        th.save({'epoch':epoch,
            'st_enc':wrap_net.net.senc.state_dict(),
            'dec':wrap_net.net.dec.state_dict(),
            'disc':wrap_net.dnet.state_dict(),
            'optimizer':optimizer.state_dict(),
            'doptimizer':doptimizer.state_dict(),
            }, save_file)
    os.chmod(save_file, 0o777)


if use_cuda:
    if len(gids) > 1:
        wrap_net = nn.DataParallel(wrap_net, device_ids=gids)
    wrap_net.cuda() #use GPU
    cudnn.benchmark = True

#============================ training & testing
debug_folder = get_debug_folder(args)
print 'debug folder:',debug_folder
if not os.path.exists(debug_folder):
    os.makedirs(debug_folder)
    os.makedirs('%s/train'%debug_folder)
    os.makedirs('%s/val'%debug_folder)
    os.chmod(debug_folder, 0o777)
    os.chmod('%s/train'%debug_folder, 0o777)
    os.chmod('%s/val'%debug_folder, 0o777)

unloader = transforms.ToPILImage()  # reconvert into PIL image
def imsave(tensor, savefile):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = unloader(image)
    image.save(savefile)
    os.chmod(savefile, 0o777)



st_iter = iter(st_trainloader)
def train(epoch):
    wrap_net.train()
    running_loss = 0.0 #generator
    running_dcl = 0.0 #discrimator classification loss
    running_dbl = 0.0 #discrimator adversarial loss
    running_time = 0.0
    total = 1e-10
    loading_time = 0.0
    end = time.time()
    ratio_thr = args.gan_ratio/max(epoch/args.gr_freq, 1.0)
    print 'epoch', epoch, 'ratio thre for GAN training', ratio_thr

    global st_iter
    for bi,(inputs, labels) in enumerate(cnt_trainloader):  #go through all content images
        if not os.path.exists(debug_folder):  #break if the debug image folder is deleted
            print 'GAN training break at mb', bi
            break
        try:
            st_inputs, st_labels = st_iter.next()
        except StopIteration:
            st_iter = iter(st_trainloader)
            st_inputs, st_labels = st_iter.next()
        if inputs.size() != st_inputs.size():
            print 'skip mb', bi, 'autoencoder training: size does not match', inputs.size(), st_inputs.size()
            continue
        if use_cuda:
            inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
            st_inputs, st_labels = st_inputs.cuda(async=True), st_labels.cuda(async=True)
        inputs,labels = Variable(inputs), Variable(labels)
        st_inputs,st_labels = Variable(st_inputs), Variable(st_labels)

        if random.random() > ratio_thr:
            use_real = True
        else:
            use_real = False #use reconstruction instead of real image to make discriminator harder

        loading_time += time.time() - end
        ###################### update discriminator  =============================
        doptimizer.zero_grad()
        if (epoch > 1 or bi > 0) and args.optm == 'padam' and (args.cla_w > 0 or args.gan_w > 0):
            doptimizer.pred_restore()  #restore weight for prediction metod

        #discriminator accuracy
        if len(gids) > 1:
            loss, running_dcl, running_dbl = wrap_net.module.train_disc(inputs, labels, st_inputs, st_labels, args, running_dcl, running_dbl, use_real=use_real)
        else:
            loss, running_dcl, running_dbl = wrap_net.train_disc(inputs, labels, st_inputs, st_labels, args, running_dcl, running_dbl, use_real=use_real)
        if loss is not None and len(loss) > 0:
            loss.backward()
            if args.optm == 'padam':
                doptimizer.pred_step()  #predition step
            else:
                doptimizer.step()
        if bi % args.print_freq == 1:
            print 'training epoch: %d, minibatch: %d, dis class loss: %.3f, dis binary loss: %.3f'%(epoch, bi, running_dcl/(bi+1), running_dbl/(bi+1))
            if len(gids) > 1:
                print 'ep%d mb%d, discriminator loss details: '%(epoch, bi), [x.data[0] for x in wrap_net.module.criterion.ls]
            else:
                print 'ep%d mb%d, discriminator loss details: '%(epoch, bi), [x.data[0] for x in wrap_net.criterion.ls]


        ##################################update generator =======================
        optimizer.zero_grad()
        if len(gids) > 1:
            tmp = wrap_net.module
        else:
            tmp = wrap_net
        for p in tmp.dnet.parameters():
            p.requires_grad = False   #not update discriminator
        unfreeze_generator(tmp.net)

        loss,img12 = wrap_net(inputs, labels, st_inputs, st_labels, args)
        if loss is not None and len(loss) > 0:
            sumloss = sum(loss)
            if len(gids) > 1:
                sumloss.backward(th.ones(len(gids)))
                running_loss += th.sum(sumloss).data[0]
            else:
                sumloss.backward()
                running_loss += sumloss.data[0]

            optimizer.step()

        running_time += time.time() - end
        end = time.time()

        if (bi % args.print_freq == 1)  or ( epoch == args.epochs and bi < 30 ) :
            print 'training epoch: %d, minibatch: %d, loss: %f,  total time/mb: %f ms, running time/mb: %f ms'%(
                    epoch, bi, running_loss/(bi+1),
                    running_time/(bi+1)*1000.0, (running_time-loading_time)/(bi+1)*1000.0)
            print 'ep%d mb%d generator loss details: '%(epoch, bi), [x.data[0] for x in loss]
            #save intermdiate results for examining
            tmp = inputs.data[0].clamp_(0, 1)
            imsave(tmp, '%s/train/e%d_b%d_%d.jpg'%(debug_folder, epoch, bi, 1))
            tmp = st_inputs.data[0].clamp_(0, 1)
            imsave(tmp, '%s/train/e%d_b%d_%d.jpg'%(debug_folder, epoch, bi, 2))
            tmp = img12.data[0].clamp_(0, 1)
            imsave(tmp, '%s/train/e%d_b%d_%d.jpg'%(debug_folder, epoch, bi, 12))

    return running_loss/len(cnt_trainloader), running_time, loading_time, running_dcl/len(cnt_trainloader), running_dbl/len(cnt_trainloader)

def test(epoch):
    wrap_net.eval()
    running_time = 0.0
    running_loss = 0.0
    total = 1e-10
    running_time = 0.0
    loading_time = 0.0
    end = time.time()
    st_titer = iter(st_testloader)
    for bi,(inputs,labels) in enumerate(cnt_testloader): #iter all content test, random select same number of style images
        try:
            st_inputs, st_labels = st_titer.next()
        except StopIteration:
            st_titer = iter(st_testloader)
            st_inputs, st_labels = st_titer.next()
        if inputs.size() != st_inputs.size():
            print 'autoencoder testing: size does not match', inputs.size(), st_inputs.size()
            continue
        if use_cuda:
            inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
            st_inputs, st_labels = st_inputs.cuda(async=True), st_labels.cuda(async=True)
        inputs,labels = Variable(inputs,volatile=True), Variable(labels,volatile=True)
        st_inputs,st_labels = Variable(st_inputs,volatile=True), Variable(st_labels,volatile=True)
        loading_time += time.time() - end

        #forward pass
        _,img12 = wrap_net(inputs, labels, st_inputs, st_labels, args)

        running_time += time.time() - end
        end = time.time()

        if bi == 1 or ( epoch == args.epochs and bi < 30 ):
            tmp = inputs.data[0].clamp_(0, 1)
            imsave(tmp, '%s/val/e%d_b%d_%d.jpg'%(debug_folder, epoch, bi, 1))
            tmp = st_inputs.data[0].clamp_(0, 1)
            imsave(tmp, '%s/val/e%d_b%d_%d.jpg'%(debug_folder, epoch, bi, 2))
            tmp = img12.data[0].clamp_(0, 1)
            imsave(tmp, '%s/val/e%d_b%d_%d.jpg'%(debug_folder, epoch, bi, 12))

    ave_time = (running_time-loading_time)/len(cnt_testloader)
    return ave_time, running_time



epoch = 0
best_file1 = get_save_file(args)
if not args.test_flag:
    print '=============================training'
    while epoch < args.epochs:
        epoch += 1
        if not os.path.exists(debug_folder):
            print 'GAN training break at epoch', epoch
            break
        print 'taining epoch', epoch
        mko.adjust_learning_rate(optimizer, args.lr, args, epoch)
        mko.adjust_learning_rate(doptimizer, args.d_lr, args, epoch)
        tr_l, running_time, loading_time, tr_dcl, tr_dbl = train(epoch)
        print '**training epoch: %d, mb: %d * %d, G loss: %f, D loss: %f, total time/mb: %f ms, running time/mb: %f ms, total time/epoch: %f s,'%(
                epoch, args.batch_size, len(cnt_trainloader), tr_l,  tr_dcl+tr_dbl,
                running_time/len(cnt_trainloader)*1000.0, (running_time-loading_time)/len(cnt_trainloader)*1000.0, running_time)
        if math.isnan(tr_l) or math.isnan(tr_dcl+tr_dbl) or math.isinf(tr_l) or math.isnan(tr_dcl+tr_dbl):
            print 'stop for abnormal GAN training'
            break
        ave_time,running_time = test(epoch)
        print '**validating epoch: %d, running time/mb: %f ms, total time/epoch: %f s'%(
                epoch,  ave_time*1000, running_time)

    save_model(epoch, wrap_net,  optimizer, doptimizer,  best_file1)
    print 'model saved to ', best_file1
    print 'training complete!'

else:
    print 'use the test script for testing!!'


print datetime.now()

