import os
import numpy as np
# import caffe
import cv2
import sys
import math
import glob
import scipy.io as sio
import time

def trans_n8(mask, px, py):
    M = np.array([[1,0,py],[0,1,px]]).astype(np.float32)
    trans = cv2.warpAffine(mask, M, mask.shape[::-1])
    return trans

def bilinear_interpolation(flux, x, y):
    xtl = int(x)
    ytl = int(y)
    x0 = x - xtl
    y0 = y - ytl
    print(x, y, x0, y0)
    xtr = max(xtl + 1, flux.shape[1]-1)
    ytr = ytl
    xbl = xtl
    ybl = max(ytl + 1, flux.shape[2]-1)
    xbr = max(xtl + 1, flux.shape[1]-1)
    ybr = max(ytl + 1, flux.shape[2]-1)
    value = np.zeros((2))
    value[0] = flux[0][xtl][ytl]*(1-x0)*(1-y0) + flux[0][xbr][ybr]*(1-x0)*y0 + flux[0][xtr][ytr]*x0*(1-y0) + flux[0][xbr][ybr]*x0*y0
    value[1] = flux[1][xtl][ytl]*(1-x0)*(1-y0) + flux[1][xbr][ybr]*(1-x0)*y0 + flux[1][xtr][ytr]*x0*(1-y0) + flux[1][xbr][ybr]*x0*y0
    print(value.shape)
    print("offset value")
    print (value[0], value[1])
    return value


def direction_to_cells_new(flux, threshold):
    magnitude, angle = cv2.cartToPolar(flux[0], flux[1])
    vmax = float(magnitude.max())    
    mask = (magnitude > threshold).astype(np.uint8)
    output = np.zeros((flux.shape[1], flux.shape[2]))

    for i in range(flux.shape[1]):
        for j in range(flux.shape[2]):
            if(mask[i][j]):
                x = i
                y = j
                for k in range(10):
                    #print("xy0", x, y)
                    #dxy = bilinear_interpolation(flux, x, y)
                    #print(flux[0][int(x)][int(y)], flux[1][int(x)][int(y)])
                    dx = flux[0][min(int(x), flux.shape[1]-1)][min(int(y), flux.shape[2]-1)]
                    dy = flux[1][min(int(x), flux.shape[1]-1)][min(int(y), flux.shape[2]-1)]
                    x += dx
                    y += dy
                    #print("xy", x, y)
                output[min(int(x), flux.shape[1]-1)][min(int(y), flux.shape[2]-1)] += 1;

                
    return mask, output




def direction_to_cells(flux, threshold):
    print(flux.shape)
    # flux = -1*flux
    # print(flux.shape)
    magnitude, angle = cv2.cartToPolar(flux[0], flux[1])
    vmax = float(magnitude.max())
    mask = (magnitude > threshold).astype(np.uint8)
    print(mask.shape)
    #mask = 255.0*mask/mask.max()
    mask_rev = (mask == 0).astype(np.uint8)
    ending = trans_n8(mask_rev, -1, -1)*mask*np.logical_and(angle >= math.pi/8, angle < 3*math.pi/8) \
           + trans_n8(mask_rev, 0, -1)*mask*np.logical_and(angle >= 3*math.pi/8, angle < 5*math.pi/8) \
           + trans_n8(mask_rev, 1, -1)*mask*np.logical_and(angle >= 5*math.pi/8, angle < 7*math.pi/8) \
           + trans_n8(mask_rev, 1, 0)*mask*np.logical_and(angle >= 7*math.pi/8, angle < 9*math.pi/8) \
           + trans_n8(mask_rev, 1, 1)*mask*np.logical_and(angle >= 9*math.pi/8, angle < 11*math.pi/8) \
           + trans_n8(mask_rev, 0, 1)*mask*np.logical_and(angle >= 11*math.pi/8, angle < 13*math.pi/8) \
           + trans_n8(mask_rev, -1, 1)*mask*np.logical_and(angle >= 13*math.pi/8, angle < 15*math.pi/8) \
           + trans_n8(mask_rev, -1, 0)*mask*np.logical_or(angle < math.pi/8, angle >= 15*math.pi/8)\
           + np.logical_and(angle >= math.pi/8, angle < 3*math.pi/8)*\
           np.logical_and(abs(angle-trans_n8(angle, 0, -1))>=3*math.pi/4, abs(angle-trans_n8(angle, 0, -1))<5*math.pi/4)*mask\
           + np.logical_and(angle >= 3*math.pi/8, angle < 5*math.pi/8) *np.logical_and(abs(angle-trans_n8(angle, 0, -1))>=3*\
            math.pi/4, abs(angle-trans_n8(angle, 0, -1))< 5*math.pi/4)*mask\
           + np.logical_and(angle >= 5*math.pi/8, angle < 7*math.pi/8) *np.logical_and(abs(angle-trans_n8(angle, 1, -1))>=3*\
            math.pi/4, abs(angle-trans_n8(angle, 1, -1))< 5*math.pi/4)*mask\
           + np.logical_and(angle >= 7*math.pi/8, angle < 9*math.pi/8) *np.logical_and(abs(angle-trans_n8(angle, 1, 0))>=3*\
            math.pi/4, abs(angle-trans_n8(angle, 1, 0))< 5*math.pi/4)*mask\
           + np.logical_and(angle >= 9*math.pi/8, angle < 11*math.pi/8) *np.logical_and(abs(angle-trans_n8(angle, 1, 1))>=3*\
            math.pi/4, abs(angle-trans_n8(angle, 1, 1))< 5*math.pi/4)*mask\
           + np.logical_and(angle >= 11*math.pi/8, angle < 13*math.pi/8) *np.logical_and(abs(angle-trans_n8(angle, 0, 1))>=3*\
            math.pi/4, abs(angle-trans_n8(angle, 0, 1))< 5*math.pi/4)*mask\
           + np.logical_and(angle >= 13*math.pi/8, angle < 15*math.pi/8) *np.logical_and(abs(angle-trans_n8(angle, -1, 1))>=3*\
            math.pi/4, abs(angle-trans_n8(angle, -1, 1))< 5*math.pi/4)*mask\
           + np.logical_or(angle < math.pi/8, angle >= 15*math.pi/8) *np.logical_and(angle-trans_n8(angle, -1, 0)>=3*\
            math.pi/4, abs(angle-trans_n8(angle, -1, 0))< 5*math.pi/4)*mask
           # + np.logical_and(angle >= math.pi/8, angle < 3*math.pi/8)*\
           # np.logical_not(abs(angle-trans_n8(angle, -1, -1))<3*math.pi/4)*mask\
           # + np.logical_and(angle >= 3*math.pi/8, angle < 5*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, 0, -1))<3*\
           #  math.pi/4)*mask\
           # + np.logical_and(angle >= 5*math.pi/8, angle < 7*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, 1, -1))<3*\
           #  math.pi/4)*mask\
           # + np.logical_and(angle >= 7*math.pi/8, angle < 9*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, 1, 0))<3*\
           #  math.pi/4)*mask\
           # + np.logical_and(angle >= 9*math.pi/8, angle < 11*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, 1, 1))<3*\
           #  math.pi/4)*mask\
           # + np.logical_and(angle >= 11*math.pi/8, angle < 13*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, 0, 1))<3*\
           #  math.pi/4)*mask\
           # + np.logical_and(angle >= 13*math.pi/8, angle < 15*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, -1, 1))<3*\
           #  math.pi/4)*mask\
           # + np.logical_or(angle < math.pi/8, angle >= 15*math.pi/8) *np.logical_not(abs(angle-trans_n8(angle, -1, 0))<3*\
           #  math.pi/4)*mask
    ending = (ending > 0).astype(np.uint8)
    
    skl = ending * (vmax - magnitude) / vmax
    
    return mask, ending

path = '../细胞数据可视化/save_MBM12/transxy/'
lists = glob.glob(path+'*.mat')
for gtpath in lists:
    flux = sio.loadmat(gtpath)['GTcls']
    direction_to_cells(flux, 0.4)
