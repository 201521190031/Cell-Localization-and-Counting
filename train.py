# coding:utf-8
import sys
import os

import warnings
from utils import save_checkpoint, get_flux

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import numpy as np
import argparse
import cv2
import dataset
import time
import math
from PIL import Image
from image import *
import scipy.io
import scipy.misc as misc
import os
import random
from hed_model import HED
from unet import U_Net
from vgg import VGG
from fpn import AutoScale
from csrnet import CSRNet
from FCN import FCN16s
from SegNet import SegNet
from BL import *
from counter import Counter, Counter_Loss
from centerloss import CenterLoss
from ssim_loss import SSIM_Loss
import gc
import matplotlib.pyplot as plt
random.seed(10)

from inference_flux import trans_n8, flux_to_skl
from find_contours import findmaxcontours

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--model_id', default=0, type=int, help='id of trained model')

warnings.filterwarnings('ignore')


def main():
    global args, best_prec1
   
    best_prec1 = 1e6
    #best_game = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-4
    args.lr =   1e-4
    lr = 1e-4
    lr_cent = 0.1
    args.rate_lr = 0.01
    args.batch_size = 1
    args.start_epoch = 0

    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 20000
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 200
    args.infer_thresh = 0.6
    # args.infer_thresh = 0.5
    # args.infer_thresh = 0.4
    args.task = "saved_model/UNet/train32_unet(AdaK+Flip)_4/"
    if not os.path.exists(args.task):
        os.mkdir(args.task)

    global trancos_train_mask, trancos_test_mask
    with open('./Cellstrain.npy', 'rb') as outfile:
        train_list = np.load(outfile).tolist()

    with open('./Cellsval.npy', 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    print(len(train_list), train_list[0],len(val_list))
    density_value = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

    # model = HED()
    model = U_Net()
    # model = FCN16s()
    # model = SegNet(3, 2)
    # model = vgg19()
    # model = CSRNet()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    print('Model:', model)

    rate_model = 1
    count_model = Counter('VGG11')
    count_model = count_model.cuda()
    # class_model = nn.DataParallel(class_model, device_ids=[1])
    # class_model = class_model.cuda(1)
    criterion0 = nn.MSELoss().cuda()
    criterion1 = Counter_Loss()

    # center_loss = CenterLoss(num_classes=1, feat_dim=1, use_gpu=True)
    # optimizer = torch.optim.Adam(
    #     [
    #         {'params': model.module.conv1.parameters(), 'lr': args.lr},
    #         {'params': model.module.conv2.parameters(), 'lr': args.lr},
    #         {'params': model.module.conv3.parameters(), 'lr': args.lr},
    #         {'params': model.module.conv4.parameters(), 'lr': args.lr},
    #         {'params': model.module.conv5.parameters(), 'lr': args.lr},
    #         {'params': model.module.cd1.parameters(), 'lr': args.lr},
    #         {'params': model.module.cd2.parameters(), 'lr': args.lr},
    #         {'params': model.module.cd3.parameters(), 'lr': args.lr},
    #         {'params': model.module.cd4.parameters(), 'lr': args.lr},
    #         {'params': model.module.cd5.parameters(), 'lr': args.lr},
    #         {'params': model.module.up2.parameters(), 'lr': args.lr},
    #         {'params': model.module.up3.parameters(), 'lr': args.lr},
    #         {'params': model.module.up4.parameters(), 'lr': args.lr},
    #         {'params': model.module.up5.parameters(), 'lr': args.lr},
    #         {'params': model.module.rd2.parameters(), 'lr': args.lr},
    #         {'params': model.module.rd3.parameters(), 'lr': args.lr},
    #         {'params': model.module.rd4.parameters(), 'lr': args.lr},
    #         {'params': model.module.rd5.parameters(), 'lr': args.lr},
    #         {'params': model.module.dsn1.parameters(), 'lr': args.lr * 0.1},
    #         {'params': model.module.dsn2.parameters(), 'lr': args.lr * 0.1},
    #         {'params': model.module.dsn3.parameters(), 'lr': args.lr * 0.1},
    #         {'params': model.module.dsn4.parameters(), 'lr': args.lr * 0.1},
    #         {'params': model.module.dsn5.parameters(), 'lr': args.lr * 0.1},
    #         {'params': model.module.dsn6.parameters(), 'lr': args.lr * 0.1},
    #     ], lr=args.lr,weight_decay=args.decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.decay)

    # count_pre = './saved_model/train32_unet(AdaK+Flip+Counter)_3/counter_best.pth.tar'
    # if os.path.isfile(count_pre):
    #     print("=> loading checkpoint '{}'".format(count_pre))
    #     checkpoint = torch.load(count_pre)

    #     model_dict = count_model.state_dict()
    #     pre_val = checkpoint['state_dict']
    #     pre_val = {k: v for k, v in pre_val.items() if k in model_dict}
    #     model_dict.update(pre_val)
    #     count_model.load_state_dict(model_dict)
    # else:
    #     print("=> no checkpoint found at '{}'".format(count_pre))
    args.pre = './saved_model/train32_unet(AdaK+Flip+Counter)/model_best_1.pth.tar'
    args.pre = None
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            model_dict = model.state_dict()
            pre_val = checkpoint['state_dict']
            pre_val = {k: v for k, v in pre_val.items() if k in model_dict}
            model_dict.update(pre_val)
            model.load_state_dict(model_dict)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    print(best_prec1)
    refine_best_mae = 10 * 500
    best_game = 6.78
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        adjust_learning_rate(optimizer, epoch)
        train(train_list, count_model, model, rate_model, criterion0, criterion1, optimizer, epoch, args.task, density_value,
            lr_cent, lr)

        end_1 = time.time()
        prec1, visi = validate(val_list, count_model, model, rate_model, criterion0, criterion1, args.task, density_value, epoch)
        end_2 = time.time()
        
        print("train time:",end_1-start,"test time:",end_2-end_1)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        print('Trained model id:',args.model_id)

        save_checkpoint({
            'epoch': epoch + 1,'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,'optimizer': optimizer.state_dict(),
        }, visi, is_best, args.task, args.model_id)
        end = time.time()


def direction_generate(img_size, k, lamda):
    gt = np.argwhere(k>0)
    # print('gt:',gt.shape)
    new_size = (int(img_size[0]*lamda),int(img_size[1]*lamda))
    skl = np.zeros(new_size)
    for o in range(0, len(gt)):
        x = int(max(1, gt[o][0]* lamda))
        y = int(max(1, gt[o][1] * lamda))
        # print(len(gt),x,y)
        if x >= new_size[0] - 1 or y >= new_size[1] - 1:
            # print(o)
            continue
        skl[x][y] = 1
    flux, mask = get_flux(skl)
    return flux, mask


def count_distance(input_pred, input_img, thr, kpoints, fname):
    mask, extract_skl = flux_to_skl(input_pred, thr)
    _, count = cv2.connectedComponents(extract_skl)

    return np.amax(count), extract_skl


def nl(input, pre_count, target, mask, k, criterion0, criterion1):
    n,c,h,w = input.size()
    # print(input.size())
    mask = np.array(mask)
    regionPos = (mask>0)
    regionNeg = (mask==0)
    sumPos = np.sum(regionPos)
    sumNeg = np.sum(regionNeg)
    weightPos = np.zeros((c, n, h, w))
    weightNeg = np.zeros((c, n, h, w))
    weight = np.zeros((c, n, h, w))
    weightPos[0] = sumNeg/float(sumPos+sumNeg)*regionPos
    weightPos[1] = sumNeg/float(sumPos+sumNeg)*regionPos
    weightNeg[0] = sumPos/float(sumPos+sumNeg)*regionNeg
    weightNeg[1] = sumPos/float(sumPos+sumNeg)*regionNeg
    weightNeg = weightNeg.transpose((1, 0, 2, 3))
    weightPos = weightPos.transpose((1, 0, 2, 3))
    # print('weightNeg:', weightNeg.shape)
    weight = np.add(weightNeg, weightPos)
    # print('weight:',weight.shape)
    weight = torch.from_numpy(weight).type(torch.FloatTensor).cuda()
    weightNeg = torch.from_numpy(weightNeg).type(torch.FloatTensor).cuda()
    weightPos = torch.from_numpy(weightPos).type(torch.FloatTensor).cuda()

    loss = torch.mean(weight*(input-target)**2) \
    # + criterion1(pre_count, k.sum())
    return loss


def save_img(img, path):
    img_save = img.mul(255).byte()
    img_save = img_save.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    im = Image.fromarray(img_save)
    im.save(path)


def train(Pre_data, counter, model, rate_model, criterion0, criterion1, optimizer, epoch, task_id, density_value, lr_cent, lr):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset_Cell(Pre_data, task_id,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]), ]),
                            train=True,
                            # seen=model.module.seen,
                            batch_size=args.batch_size,num_workers=args.workers),
        batch_size=args.batch_size,drop_last=False)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    # classifier.eval()
    end = time.time()
    log_data_train = []

    all_count_gt = 0
    all_count_knn = 0

    # scale_map = [0.8,0.85,0.9,0.95,1.0,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50]
    for i, (img, target, k, fname, mask, img_raw) in enumerate(train_loader):
        start  = time.time()
        # imgs = imgs.squeeze(0)
        # imgs = imgs.cuda(1)
        # targets = targets.squeeze(0)
        # masks = masks.squeeze(0)
        # ks = ks.squeeze(0)
        # ks = np.array(ks)
        # img_raws = img_raws.squeeze(0)
        # img_raws = np.array(img_raws)
        # # print(imgs.shape)

        # sizes = torch.max(classifier(imgs),1)[1]
        # j = 0

        # for img, target, k, mask, img_raw, size in zip(imgs, targets, ks, masks, img_raws, sizes):
        k = k.cuda()
        
        loss = 0
        # img = img.unsqueeze(0)
        img = img.cuda()
        # print('img:',img.shape)

        rate = 2
        # img_size = (img.shape[2], img.shape[3])
        mask = np.asarray(mask)
        # print('mask:',mask.shape)
        # img_size = img.size()[2:]
        # if size == 1:
        target =  target.type(torch.FloatTensor).cuda()
        # print("largeimg shape:{}, largetarget shape:{}".format(largeimg.shape, largeflux.shape))
        
        # Hed_result_0, Hed_result_1, Hed_result_2, Hed_result_3, Hed_result_4, Hed_result_5 = model(img, target,
        # 
        # U_Net                                                                             refine_flag=False)
        Hed_result_5 = model(img, target)[0]
        # #FCN, SegNet, BL
        # Hed_result_5 = model(img)
        # #CSRNet
        # Hed_result_5 = model(img, target)
        # Hed_result_5_Cat = torch.cat((Hed_result_5, img), 1)
        # print('Hed_result_5 shape:',Hed_result_5.shape)
        # pre_count = counter(Hed_result_5_Cat)
        pre_count = 0
        #save_img(img,"1.jpg")
        #print(Hed_result_0.shape, target.shape)
        end_1  = time.time()
        #target = target.unsqueeze(0)
        # print("Hed_result",Hed_result_0.shape)
        # print(Hed_result_1.shape)
        # print(Hed_result_2.shape)
        # print(Hed_result_3.shape)
        # print(Hed_result_4.shape)
        # print(Hed_result_5.shape)
        # print('target:',target.shape)

        # loss +=  nl(Hed_result_0, target, mask, criterion)\
        # +nl(Hed_result_1, target, mask, criterion)\
        # +nl(Hed_result_2, target, mask, criterion)\
        # +nl(Hed_result_3, target, mask, criterion)\
        # +nl(Hed_result_4, target, mask, criterion)\
        # +nl(Hed_result_5, target, mask, criterion)\

        loss += nl(Hed_result_5, pre_count, target, mask, k, criterion0, criterion1)  
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # img_show = img_raw.numpy().squeeze(0).copy()
        # # target_show = ((target).cpu()).numpy().copy()
        # # target_show = np.array(255.0*(target_show/target_show.max()), dtype=np.float32)
        # # print(img_show.shape, target_show.shape)
        # # cv2.imwrite('img_train.png', img_show)
        # cv2.imwrite('train_data_show/'+fname[0].replace('.h5', 'IMG.png'), img_show)
        # show_map = Hed_result_5.detach().cpu().numpy()
        # show_map = np.squeeze(show_map, 0)
        # magnitude, angle = cv2.cartToPolar(show_map[0], show_map[1])
        # show_map = 255.0*magnitude/magnitude.max()
        # cv2.imwrite('train_data_show/'+fname[0].replace('.h5', 'Pred.png'), show_map)
        # show_gt = mask.copy()
        # show_gt = show_gt.squeeze(0)
        # # show_gt = np.squeeze(show_gt, 0)
        # # print("gt_shape: ",show_gt.shape)
        # show_gt = 255.0*show_gt/show_gt.max()
        # cv2.imwrite('train_data_show/'+fname[0].replace('.h5', 'GT.png'), show_gt)
        data_time.update(time.time() - end)
        end_2 = time.time()
        # j+= 1

        if i % args.print_freq == 0:

            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            with open('./logs/train_logs/trainlog'+str(args.model_id)+'.txt', 'a+') as f:
                f.write('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t\n'
                .format(
                epoch+1, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            #print(fname)
        

def validate(Pre_data, counter, model, rate_model, criterion0, criterion1, task_id, density_value, epoch):
    print('begin test')
    # Pre_data = pre_data(val_list,train=False)

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_Cell(Pre_data, task_id,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),batch_size=1)

    model.eval()
    # classifier.eval()

    mae = 0.0
    mse = 0.0
    original_mae = 0
    visi = []
    error = []

    log_data_test = []
    scale_map = [1]

    list_class = []

    Gmae = 0
    # Rsize  = 0
    for i, (img, target, k, fname, mask, img_raw) in enumerate(test_loader):
        # print(fname)
        start = time.time()
        pre_count = 0
        # imgs = imgs.cuda(1)
        
        # rate = 2
        # imgs = imgs.squeeze(0)
        # targets = targets.squeeze(0)
        # ks = ks.squeeze(0)
        # ks = np.array(ks)
        # masks  = masks.squeeze(0)
        # img_raws = img_raws.squeeze(0)
        # img_raws = np.array(img_raws)

        # sizes = torch.max(classifier(imgs),1)[1]
        # # sizes =  sizes.detach().cpu().numpy()
        # # print('size:',sizes)
        # j = 0
        # for img, target, k, mask, img_raw, size in zip(imgs, targets, ks, masks, img_raws, sizes):
        #     img = img.unsqueeze(0)
        #     img_size = (img.shape[2], img.shape[3])
        mask = np.asarray(mask)
        k = k.numpy()
        # if size == 1:
        #     Rsize += 1
        #     img = F.upsample_bilinear(img, (int(img.size()[2] * rate), int(img.size()[3] * rate)))
        #     img_raw = cv2.resize(img_raw,(int(img_raw.shape[0]*rate), int(img_raw.shape[1]*rate)))
        #     target, mask = direction_generate(img_size, k, rate)
        #     target = (torch.from_numpy(target))
        target =  target.type(torch.FloatTensor).cuda()
        img = img.cuda()

        # print("img shape:{}, map_gt shape:{}".format(img.shape, target.shape))
        # cv2.imwrite('val_data_show/'+fname[0].replace('.h5', 'GT.png'), target_show)

        # Hed_result_5 = model(img, target, refine_flag=False)[5]
        #UNet
        Hed_result_5 = model(img, target)[0]
        # #CSRNet
        # Hed_result_5 = model(img, target)
        # #FCN, SegNet, BL
        # Hed_result_5 = model(img)
        original_distance_map = Hed_result_5.detach().cpu().numpy()
        show_map = original_distance_map.squeeze(0)
        pre_count, skl = count_distance(show_map, img_raw, args.infer_thresh, k, fname)
        # pred_count += pre_count

        # img_show = img_raw.numpy().squeeze(0).copy()
        # target_show = ((target).cpu()).numpy().copy()
        # target_show = np.array(255.0*(target_show/target_show.max()), dtype=np.float32)
        # print(img_show.shape, target_show.shape)
        # cv2.imwrite('val_data_show/'+fname[0].replace('.h5', 'IMG.png'), img_show)
        # magnitude, angle = cv2.cartToPolar(show_map[0], show_map[1])
        # print('show mask max:', magnitude.max())
        # # print('show mask max:', magnitude.max())
        # # show_map = np.squeeze(show_map, 0)
        # mag_thresh = (magnitude>args.infer_thresh).astype(np.uint8)
        # show_map = 255.0*magnitude/magnitude.max()
        # # mask = mask.squeeze(0)
        # cv2.imwrite('val_data_show/'+fname[0].replace('.h5', 'Pred.png'), show_map)
        # show_gt = mask.copy()
        # show_gt = np.squeeze(show_gt, 0)
        # show_gt = 255.0*show_gt/show_gt.max()
        # cv2.imwrite('val_data_show/'+fname[0].replace('.h5', 'GT.png'), show_gt)
        # j += 1
            # skl = 255.0*skl/skl.max()
            # cv2.imwrite('skeleton.jpg', skl)
        Gt_count = k.sum()
        mae+= abs(pre_count - Gt_count)
        mse += abs(pre_count - Gt_count) * abs(pre_count - Gt_count)
        error.append(pre_count - Gt_count)
        end = time.time()

        #show_map = original_distance_map.squeeze(0)
        
        if i % 32 == 0:
            # print(pre_count_crop.shape,distance_map_gt.shape)
            skl = 255.0*skl/skl.max()
            visi.append([img.data.cpu().numpy(), skl,
                         target.unsqueeze(0).data.cpu().numpy(), fname])
            print(
                i, fname[0], "Gt:", Gt_count, "pre_count:", pre_count, mae/(i+1), Gmae/(i+1))
        # print("all",end-start,"loca:",end_1-start_1)

    mae = mae/len(test_loader)
    Gmae = Gmae/len(test_loader)
    mse = math.sqrt(mse/len(test_loader))
    original_mae = original_mae / len(test_loader)
    # print('Rsize:',Rsize)

    with open('./logs/val_logs/vallog'+str(args.model_id)+'.txt', 'a+') as f:
        f.write('Epoch:{epoch:d} * MAE {mae:.3f} ,* MSE {mse:.3f}\n*ERROR {error:}\n'
            .format(epoch=epoch+1, mae=mae, mse=mse, error = error))
        print('Epoch:{epoch:d} * MAE {mae:.3f} ,* MSE {mse:.3f}\n*ERROR {error:}'
            .format(epoch=epoch+1, mae=mae, mse=mse, error= error))
    # print("original_mae",original_mae)

    return mae, visi

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
