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
import  cv2
# get_ipython().magic(u'matplotlib inline')

img_train_path = '../datasets/cells/data/'
img_test_path = '../datasets/cells/data/'
gt_train_path = '../datasets/cells/mat_gt/'
gt_test_path = '../datasets/cells/mat_gt/'
path_sets = [img_train_path, img_test_path]
# save_train_path = '../datasets/cells/train_data/'
# save_test_path = '../datasets/cells/test_data/'
#save_train_path_gt = '../datasets/cells/train_gt'

#set the root to the Shanghai dataset you download
# root = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/'


# #now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root,'part_A_final/train_data','images')
# part_A_test = os.path.join(root,'part_A_final/test_data','images')
# # part_A_train = os.path.join(root,'part_A_final/train_data','images')
# # part_A_test = os.path.join(root,'part_A_final/test_data','images')
# path_sets = [part_A_train,part_A_test]
# save_path_A_train = os.path.join(root,'part_A_final/train_data','ground_truth_original')
# save_path_A_test = os.path.join(root,'part_A_final/test_data','ground_truth_original')
# if not os.path.exists(save_path_A_train):
#     os.mkdir(save_path_A_train)
# if not os.path.exists(save_path_A_test):
#     os.mkdir(save_path_A_test)

# save_path = './save_image/'

# save_test_path = '../datasets/cells/save_image_test/'
save_path = '../datasets/cells/save_image_density/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# if not os.path.exists(save_test_path):
#     os.mkdir(save_test_path)



img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*cell.png')):
        img_paths.append(img_path)
#img_paths = [img_path for img_path in img_paths if not img_path.startswith('.')]
print(img_paths)


for img_path in img_paths:
    print(img_path.replace('cell.png','dots.mat').replace('/data/','/mat_gt/'))

    mat = io.loadmat(img_path.replace('cell.png','dots.mat').replace('/data/','/mat_gt/'))
    img = cv2.imread(img_path)
    rate = 1.5
    # img = cv2.resize(img,(0,0),fx= rate, fy=rate,interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img,(0,0),fx= 2, fy=2,interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(img_path.replace('images','images_x2_bicubic').replace('jpg','jpg'),img)

    img= plt.imread(img_path)
    k = np.zeros((int(img.shape[0]*rate), int(img.shape[1]*rate)))
    gt = np.array(mat['coordinate'])
    gt = gt * rate


    for i in range(0,len(gt)):
        if int(gt[i][1])<int(img.shape[0]*rate) and int(gt[i][0])<int(img.shape[1]*rate):
            # print(gt[i][1],gt[i][0])
            k[int(gt[i][0]),int(gt[i][1])]=1

    kpoint = k.copy()
    k = gaussian_filter(k,  2)
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



    with h5py.File(os.path.join(save_path, img_path.split('/')[-1].replace('cell.png','dots.h5')), 'w') as hf:
            hf['density_ori'] = k
            hf['kpoint'] = kpoint
            hf['density_map'] = k

    k = 255.0*(k/np.max(k))

    cv2.imwrite(os.path.join(save_path, img_path.split('/')[-1].replace('cell.png','dots.jpg')), k)
    

print ("end")