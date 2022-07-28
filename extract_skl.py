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
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import numpy as np
import h5py
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
from fpn import AutoScale
from centerloss import CenterLoss
import gc
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import cm

from find_contours import findmaxcontours
from inference_flux import trans_n8, flux_to_skl

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--model_id', default=0, type=int, help='id of trained model')
warnings.filterwarnings('ignore')
# g_save = '../datasets/MBM_data/magnitude_and_angle'
# if not os.path.exists(g_save):
#     os.mkdir(g_save)

def main():
    global args, best_prec1
   
    best_prec1 = 1e6
    args = parser.parse_args()
    args.result = "../datasets/MBM_data/source"
    args.infer_thresh = 0.4

    if not os.path.exists(args.result):
        os.mkdir(args.result)
    val_list = glob.glob(args.result+'/*.png')


    # precisions, recalls = validate(val_list, args.infer_thresh)
    validate(val_list, args.infer_thresh)
    # print(' * best MAE {mae:.3f} '
    #           .format(mae=best_prec1))
    # precisions = np.array(precisions, dtype=np.float32)
    # recalls = np.array(recalls, dtype=np.float32)
    # recalls = recalls.mean()*100.0
    # precisions = precisions.mean()*100.0
    # with open('save_result/result.txt', 'a+') as f:
    #     f.write('Epoch:{epoch:d} * Precision {precision:.3f} ,* Recall {recall:.3f}\n'
    #         .format(epoch=epoch+1, precision=precisions, recall=recalls))
    #     print('Epoch:{epoch:d} * Precision {precision:.3f} ,* Recall {recall:.3f}\n'
    #         .format(epoch=epoch+1, precision=precisions, recall=recalls))


def count_distance( input_pred, fname, thresh):
    # input_image = input_image.squeeze(0)
    # gt = gt.numpy()
    # gt = np.squeeze(gt)
    # gt_loc = np.array(list(np.where(gt==1))).transpose((1,0))
    # input_pred = input_pred.squeeze(0)

    #模长和骨架提取
    input_pred = np.array(input_pred)
    M = np.array([[1,0,-1],[0,1,-1]]).astype(np.float32)
    # print('core')
    trans = cv2.warpAffine(input_pred, M, input_pred.shape[:-1])
    # ending = trans_n8(input_pred, -1, -1)
    # mask, extract_skl = flux_to_skl(input_pred, thresh)
    # #cv2.imwrite('./local_minimum_save/distance_best_refine.pgm', extract_skl)
    # magnitude, angle = cv2.cartToPolar(input_pred[0], input_pred[1])
    # magnitude[magnitude>1]=1

    #模长和角度预测的可视化
    # fig = plt.figure('predict_'+fname[0].split('.h5')[0], figsize=(15,5))
    # ax0 = fig.add_subplot(131)
    # ax0.set_title('Image')
    # ax0.set_autoscale_on(True)
    # im0 = ax0.imshow(input_image,cmap=cm.jet)
    # ax1 = fig.add_subplot(132)
    # ax1.set_title('Norm_Pred')
    # ax1.set_autoscale_on(True)
    # im1 = ax1.imshow(magnitude,cmap=cm.jet)
    # plt.colorbar(im1,shrink=0.9)
    # ax2 = fig.add_subplot(133)
    # ax2.set_title('Angle_Pred')
    # ax2.set_autoscale_on(True)
    # im2 = ax2.imshow(angle,cmap=cm.jet)
    # plt.colorbar(im2,shrink=0.9)
    # plt.savefig(g_save+'/'+fname[0].replace('.h5', '.png'))

    #局部极小点提取
    # f = os.popen('./extract_local_minimum_return_xy ./local_minimum_save/distance_best_refine.pgm 0 ./local_minimum_save/distance_best_refine.pgm local_minimum_save/'+fname[0].split('.h5')[0]+'.pp', 'r')
    # count = f.readlines()
    # count_pred = len(np.where(input_pred==0)[0])
    # count = float(count[0].split('=')[1])
    # locsf = open('pos_minima.txt', 'r')
    # local_map = cv2.imread('local_minimum_save/'+fname[0].split('.h5')[0]+'.pp')
    # locs = locsf.readlines()
    # localizations =  np.array(list(np.where(local_map==255))).transpose((1,0))

    # localizations = np.array(localizations)
    # _, _, precision = caculate_precision_baseline(input_image, gt_loc, (600, 600), True, fname, localizations)
    # _, _,recall = caculate_recall_baseline(input_image, gt_loc, (600, 600), True, None, localizations)
    # print(precision)

    # return count, precision, recall, count_pred
    _, count = cv2.connectedComponents(extract_skl)
    return np.amax(count)


def validate(Pre_data, thresh):
    print('begin test')

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
    for i,  fname in enumerate(Pre_data):

        print(fname)
        original_distance_map = cv2.imread(fname)
        # f = h5py.File( fname)
        # original_distance_map = f['flux']
        # k = f['gt']
        
                    
        #PR计算和细胞计数
        # pre_count, precision, recall, count_pred = count_distance(img_raw.cpu().numpy(), original_distance_map, k, fname)
        pre_count = count_distance(original_distance_map,  fname, thresh)
        # precisions.append(precision)
        # recalls.append(recall)

        #Gt_count = torch.sum(k)

    #     mae.append(abs(pre_count - Gt_count))
    #     error.append(pre_count - Gt_count)
    #     mse.append(abs(pre_count - Gt_count) * abs(pre_count - Gt_count))

    #     end = time.time()
    #     visi.append([img.data.cpu().numpy(), original_distance_map,
    #                 target.unsqueeze(0).data.cpu().numpy(), fname])

    # mae = np.array(mae)
    # Gmae = Gmae/len(Pre_data)
    # mse =np.array(mse)
    # original_mae = original_mae / len(Pre_data)
    # error = np.array(error)

    # original_mae_pred = original_mae_pred / len(Pre_data)

    # with open('vallog10.txt', 'a+') as f:
    #     wstr = 'Model_id:{id:d},  Epoch:{epoch:d} * MAE {mae:.3f} ,* MSE {mse:.3f}, * STD {std:.3f}'.format(id=args.model_id, epoch=epoch+1, mae=np.mean(mae), mse=math.sqrt(mse.mean()), std = np.std(mae))
    #     f.write(wstr+'\n')
    #     f.write('Epoch:{}, * MAE {} \n* MSE {}\n *ERROR {}\n'.format(epoch+1, mae, mse, error))
    #     print('Epoch:{}, * MAE {} \n* MSE {}\n *ERROR {}\n'.format(epoch+1, mae, mse, error))
    #     print(wstr)

    # return mae, visi, precisions, recalls


if __name__ == '__main__':
    main()