import h5py
# import torch
import shutil
import numpy as np
import cv2
import os
import scipy.io
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import cm
from PR_Hungarian import *


# #MonuSeg
# g_threshold = 8
#CRC
g_threshold = 6
# #MBM
# g_threshold = 6
# #Cells
# g_threshold = 6


def generate_pred_gt(result_dir, input_pred, gt, idx, thresh):

    gt = gt.tolist()

    #得到预测细胞连通区域
    contours, _ = cv2.findContours(input_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    record_pred = open(result_dir+'pred_loc('+str(thresh)+').txt', 'a+')
    ids_pred = []

    record_gt = open(result_dir+'gt_loc('+str(thresh)+').txt', 'a+')
    ids_gt = []

    #get point, thresh, and level from gt
    for i in range(len(gt)):
        gt_x = gt[i][1]
        gt_y = gt[i][0]
        min_distance = 1e8
        ids_gt.append([gt_x, gt_y, g_threshold, g_threshold, 0])


    #get point from pred
    for j, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        center_x = int(x + w//2)
        center_y = int(y + h//2)
        ids_pred.append([center_x, center_y])

    ids_pred = np.array(ids_pred)
    ids_gt = np.array(ids_gt)

    # write loc to pred
    loc_str_pred = ''
    for i_id in range(ids_pred.shape[0]):
        loc_str_pred = loc_str_pred + ' ' + str(ids_pred[i_id][0]) + ' ' + str(ids_pred[i_id][1]) # x, y

    pred = ids_pred.shape[0]

    # write loc to pred
    print(f'{idx} {pred:d}{loc_str_pred}', file=record_pred)
    print(f'{idx} {pred:d}')
    record_pred.close()

    # write point, thresh, and level to gt
    loc_str_gt = ''
    for i_id in range(ids_gt.shape[0]):
        loc_str_gt = loc_str_gt + ' ' + str(ids_gt[i_id][0]) + ' ' + str(ids_gt[i_id][1])\
        + ' ' + str(ids_gt[i_id][2]) + ' ' + str(ids_gt[i_id][3]) + ' ' + str(ids_gt[i_id][4]) # x, y, thresh1, thresh2, level

    gt_len = ids_gt.shape[0]

    # write point, thresh, and level to gt
    print(f'{idx} {gt_len:d}{loc_str_gt}', file=record_gt)
    print(f'{idx} {gt_len:d}')
    record_gt.close()

    
    return  

