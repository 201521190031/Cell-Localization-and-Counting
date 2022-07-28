
# coding: utf-8

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial
import scipy
import json
from matplotlib import cm as CM

import torch
import  cv2
# get_ipython().magic(u'matplotlib inline')


#set the root to the Shanghai dataset you download
root = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/'


#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
# part_A_train = os.path.join(root,'part_A_final/train_data','images')
# part_A_test = os.path.join(root,'part_A_final/test_data','images')
path_sets = [part_A_train,part_A_test]
save_path_A_train = os.path.join(root,'part_A_final/train_data','ground_truth_original')
save_path_A_test = os.path.join(root,'part_A_final/test_data','ground_truth_original')
if not os.path.exists(save_path_A_train):
    os.mkdir(save_path_A_train)
if not os.path.exists(save_path_A_test):
    os.mkdir(save_path_A_test)


save_path = './save_image/'


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


for img_path in img_paths:

    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = cv2.imread(img_path)
    rate = 1
    # img = cv2.resize(img,(0,0),fx= rate, fy=rate,interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img,(0,0),fx= 2, fy=2,interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(img_path.replace('images','images_x2_bicubic').replace('jpg','jpg'),img)

    img= plt.imread(img_path)
    k = np.zeros((img.shape[0] ,img.shape[1] ))
    gt = mat["image_info"][0][0][0][0][0]
    gt = gt * rate


    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            # print(gt[i][1],gt[i][0])
            k[int(gt[i][1]),int(gt[i][0])]=1

    kpoint = k.copy()
    k = gaussian_filter(k,  8)
    print(img_path, k.sum(),len(gt),img.shape,k.shape)
    '''generate sigma'''
    pts = np.array(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0]))
    leafsize = 2048
    # build kdtree

    # tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # # query kdtree
    # distances, locations = tree.query(pts, k=2)
    # sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
    # for i, pt in enumerate(pts):
    #     sigma = (distances[i][1]) / 2
    #     sigma_map[pt[1], pt[0]] = sigma



    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth_original'), 'w') as hf:
            hf['density_ori'] = k
            hf['kpoint'] = kpoint
            hf['density_map'] = k

print ("end")