# coding:utf-8
import sys
import os

import warnings

#from pool_model import CSRNet
from crop_half import crop_half
from utils import save_checkpoint, save_true_pred, caculate_precision_baseline, caculate_recall_baseline

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import numpy as np
import argparse
import json
import cv2
import dataset
import time
import math
from PIL import Image
from image import *
import scipy.io
import scipy.misc as misc
import os
from hed_model import HED
from rate_model import RATEnet
from unet import U_Net
from centerloss import CenterLoss
import gc
import matplotlib.pyplot as plt

from find_contours import findmaxcontours

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
warnings.filterwarnings('ignore')

def main():
    global args, best_prec1
   
    best_prec1 = 1e6
    #best_game = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-4
    args.lr =  2 *1e-4
    #lr = 1e-4
    #lr_cent = 0.1
    args.rate_lr = 0.01
    args.batch_size = 1
    args.start_epoch = 0

    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 1
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 200
    args.task = "save_file_baseline_trancos"
    args.result = "save_result_cross_MBM(FPN64)"
    if not os.path.exists(args.task):
        os.mkdir(args.task)

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    global trancos_train_mask, trancos_test_mask

    with open('../Flux_MBM/Cellstest.npy', 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    # with open('./Cellstest.npy', 'rb') as outfile:
    #     val_list = np.load(outfile).tolist()
    print(val_list)
    print(len(val_list), val_list[0],len(val_list))

    # center_list = []
    # with open('./CellsCenter.npy', 'rb') as outfile:
    #     center_list = np.load(outfile).tolist()

    # density_value = args.density_value
    density_value = 3
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # model = CSRNet()
    model = HED()
    # model = U_Net()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    print(model)

    # rate_model = RATEnet()
    # rate_model = nn.DataParallel(rate_model, device_ids=[0])
    # rate_model = rate_model.cuda()
    rate_model = 1
    # criterion = nn.MSELoss(size_average=False).cuda()

    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    args.pre = './saved_model/train64/model_best.pth.tar'
    #args.pre = None
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            #args.start_epoch = checkpoint['epoch']
            args.start_epoch = 0
            best_prec1 = checkpoint['best_prec1']

            model_dict = model.state_dict()
            pre_val = checkpoint['state_dict']
            # print(pre_val.items())
            pre_val = {k: v for k, v in pre_val.items() if k in model_dict}
            model_dict.update(pre_val)
            model.load_state_dict(model_dict)

            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.epochs):
        start = time.time()
        #adjust_learning_rate(optimizer, epoch)
        start = time.time()
        #train(train_list, model, rate_model, criterion, center_loss, optimizer, epoch, args.task, density_value,
        #     lr_cent, lr)

        end_1 = time.time()
        # prec1, visi, precisions, recalls = validate(val_list, center_list, model, rate_model, criterion, args.task, density_value, epoch)
        validate(val_list, model, rate_model, criterion, args.task, density_value, epoch)
        end_2 = time.time()
        # print("train time:",end_1-start,"test time:",end_2-end_1)
        # print(' * best MAE {mae:.3f} '
        #       .format(mae=best_prec1))
        # #print("best game1:", best_game)
        # save_true_pred(visi, args.task)
        # end = time.time()
        # precisions = np.array(precisions, dtype=np.float32)
        # recalls = np.array(recalls, dtype=np.float32)
        # recalls = recalls.mean()*100.0
        # precisions = precisions.mean()*100.0
        # with open('save_result/result.txt', 'a+') as f:
        #     f.write('Epoch:{epoch:d} * Precision {precision:.3f} ,* Recall {recall:.3f}\n'
        #         .format(epoch=epoch+1, precision=precisions, recall=recalls))
        #     print('Epoch:{epoch:d} * Precision {precision:.3f} ,* Recall {recall:.3f}\n'
        #         .format(epoch=epoch+1, precision=precisions, recall=recalls))


def count_distance(input_image, input_pred, kpoint, fname):
    input_pred = input_pred.squeeze(0).squeeze(0)
    input_image = input_image.squeeze(0)
    _, pred_thresh = cv2.threshold(input_pred, 0.004,255, cv2.THRESH_BINARY)
    pred_thresh = np.array(pred_thresh, dtype= np.uint8)
    
    gt = np.array(list(np.where(kpoint==1))).transpose(1, 0)
    #print(gt)
    #print(len(locs))
    precision = caculate_precision_baseline(input_image, pred_thresh, gt, (256, 256), fname)
    recall = caculate_recall_baseline(input_image, pred_thresh, gt, (256, 256), None)
    print(precision)


    return precision, recall


def distance_generate(img_size, k, lamda, crop_size):
    #distance = max(1.0, 1.0 * lamda)
    time_start = time.time()


    distance = 1.0
    new_size = [0, 1]

    new_size[0] = img_size[2] * lamda
    new_size[1] = img_size[3] * lamda

    d_map = (np.zeros([int(new_size[0]), int(new_size[1])]) + 255).astype(np.uint8)
    gt = np.nonzero(k)

    if len(gt) == 0:
        distance_map = np.zeros([int(new_size[0]), int(new_size[1])])
        distance_map[:, :] = 10
        return new_size, distance_map

    # print(k,gt_data,gt,lamda)
    for o in range(0, len(gt)):
        x = int(max(1, gt[o][1].numpy() * lamda))
        y = int(max(1, gt[o][2].numpy() * lamda))
        # print(len(gt),x,y)
        if x >= new_size[0] - 1 or y >= new_size[1] - 1:
            # print(o)
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)
    #distance_mask = distance_map.copy()

    distance_map[(distance_map >= 0) & (distance_map < 1)] = 0
    distance_map[(distance_map >= 1) & (distance_map < 2)] = 1
    distance_map[(distance_map >= 2) & (distance_map < 3)] = 2
    distance_map[(distance_map >= 3) & (distance_map < 4)] = 3
    distance_map[(distance_map >= 4) & (distance_map < 5 * distance)] = 4
    distance_map[(distance_map >= 5 * distance) & (distance_map < 6 * distance)] = 5
    distance_map[(distance_map >= 6 * distance) & (distance_map < 8 * distance)] = 6
    distance_map[(distance_map >= 8 * distance) & (distance_map < 12 * distance)] = 7
    distance_map[(distance_map >= 12 * distance) & (distance_map < 18 * distance)] = 8
    distance_map[(distance_map >= 18 * distance) & (distance_map < 28 * distance)] = 9
    distance_map[(distance_map >= 28 * distance)] = 10

    time_end = time.time()
    #print('time cost', time_end - time_start, 's')
    #mask, distance_map = mask_generate(distance_map,distance_mask)
    # x = int(crop_size[0]*lamda)
    # y = int(crop_size[1]*lamda)
    # w = int(crop_size[2]*lamda)
    # h = int(crop_size[3]*lamda)

    # distance_map = distance_map[y:(y + h), x:(x + w)]

    # Distance_map = distance_map / np.max(distance_map) * 255
    # cv2.imwrite("1_dis.jpg",Distance_map)

    return new_size, distance_map


def validate(Pre_data, model, rate_model, criterion, task_id, density_value, epoch):
    print('begin test')
    # Pre_data = pre_data(val_list,train=False)

    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset_Cell_Val(Pre_data,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),batch_size=1)

    model.eval()

    mae = []
    mse = []
    error = []
    mae_pred = []
    mse_pred = []
    original_mae = 0
    original_mae_pred = 0
    visi = []

    Gmae = 0
    precisions = []
    recalls = []
    for i, (img, target, k, fname, img_raw) in enumerate(test_loader):
        start = time.time()
        print(fname)
        pre_count = 0
        img = img.cuda()
        #mask = mask.type(torch.FloatTensor).cuda()

        #img = img * mask
        target = target.cuda()
        rate = 1.0
        img_size = img.size()
        # img = F.upsample_bilinear(img, (int(img.size()[2] * rate), int(img.size()[3] * rate)))
        # distance_map_gt = distance_generate(img_size, k, rate, [0, 0, 0, 0])[1]
        # distance_map_gt = torch.from_numpy(distance_map_gt).unsqueeze(0).type(torch.LongTensor).cuda()

        # Hed_result_5 = model(img, target)[0]
        Hed_result_5 = model(img, target, refine_flag=False)[5]
        
        # centers = torch.squeeze(centers, 0)
        # centers = centers.numpy()
        #Hed_result_5 = Hed_result_5*mask
        original_distance_map = Hed_result_5.detach().cpu().numpy()
        original_distance_map = original_distance_map.squeeze(0)
        with h5py.File(os.path.join(args.result, fname[0]), 'w') as f:
            f['flux'] = original_distance_map
            f['gt'] = k
        # precision, recall = count_distance(img_raw.cpu().numpy(), original_distance_map, k, fname)
        # precisions.append(precision)
        # recalls.append(recall)

        #Gmae += GAME(k, fname)
        #print(pre_0.shape,  pre_1.shape, pre_up.shape, original_distance_map.shape, target.shape)
        # pre_count = original_distance_map.sum()
        Gt_count = torch.sum(k)

        mae.append(abs(pre_count - Gt_count))
        error.append(pre_count - Gt_count)
        mse.append(abs(pre_count - Gt_count) * abs(pre_count - Gt_count))

    #     # mae_pred.append(abs(count_pred - Gt_count))
    #     # mse_pred.append(abs(count_pred - Gt_count) * abs(count_pred - Gt_count))
    #     end = time.time()
    #     # print(pre_count_crop.shape,distance_map_gt.shape)
    #     visi.append([img.detach().cpu().numpy(), original_distance_map,
    #                 target.unsqueeze(0).detach().data.cpu().numpy(), fname])

    # mae = np.array(mae)
    # Gmae = Gmae/len(test_loader)
    # mse =np.array(mse)
    # original_mae = original_mae / len(test_loader)
    # error = np.array(error)

    # # mae_pred = np.array(mae_pred)
    # # #Gmae = Gmae/len(test_loader)
    # # mse_pred = np.array(mse_pred)
    # original_mae_pred = original_mae_pred / len(test_loader)

    # # with open('./Cells64(UNet)_cross_MBM_vallog.txt', 'a+') as f:
    # with open('./Cells64(FPN)_vallog.txt', 'a+') as f:
    #     wstr = 'Epoch:{epoch:d} * MAE {mae:.3f} ,* MSE {mse:.3f}, * STD {std:.3f}'.format(epoch=epoch+1, mae=np.mean(mae), mse=math.sqrt(mse.mean()), std = np.std(mae))
    #     f.write(wstr+'\n')
    #     f.write('Epoch:{}, * MAE {} \n* MSE {}\n *ERROR {}\n'.format(epoch+1, mae, mse, error))
    #     print(wstr)

    # # with open('./vallog.txt', 'a+') as f:
    # #     f.write('Epoch:{epoch:d} * MAE {mae:.3f} ,* MSE {mse:.3f}, * STD {std:.3f}\n'
    # #         .format(epoch=epoch+1, mae=np.mean(mae_pred), mse=math.sqrt(mse_pred.mean()), std = np.std(mae_pred)))
    # #     print('Epoch:{epoch:d} * MAE {mae:.3f} ,* MSE {mse:.3f}, * STD {std:.3f}'
    # #         .format(epoch=epoch+1, mae=np.mean(mae_pred), mse=math.sqrt(mse_pred.mean()), std = np.std(mae_pred)))
    # # print("original_mae",original_mae)

    # return mae, visi, precisions, recalls


if __name__ == '__main__':
    main()