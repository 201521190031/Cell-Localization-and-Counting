import numpy as np
import cv2
import math
import scipy.io
import os
import h5py
import scipy.misc
import os

# img_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/'
# gt_train_path = './part_A_final/train_data/ground_truth/'
# save_train_path = './part_A_final_25/train_data/'


img_train_path = '../datasets/MBM_data/source/'
gt_train_path = '../datasets/MBM_data/annotations/'
save_train_path = '../datasets/MBM_data/annotations/'
save_vis_path = '../datasets/MBM_data/VisGt/'
if not os.path.exists(save_train_path):
    os.mkdir(save_train_path)
if not os.path.exists(save_vis_path):
    os.mkdir(save_vis_path)
# if not os.path.exists(save_path_A_test):
#     os.mkdir(save_path_A_test)

img_train = []
gt_train = []

for file_name in os.listdir(img_train_path):
    if file_name.endswith('.png'):
        img_train.append(file_name)

for file_name in os.listdir(gt_train_path):
    if file_name.endswith('.png'):
        gt_train.append(file_name)

distance = 1
img_train.sort()
gt_train.sort()
print(gt_train)
sumdot = []

for k in range(len(img_train)):

    Img_data = cv2.imread(img_train_path + img_train[k])
    Gt_data = cv2.imread(gt_train_path + gt_train[k])
    #print(Gt_data.shape)
    Gt_data = cv2.cvtColor(Gt_data, cv2.COLOR_BGR2GRAY)
    Gt_data = np.array(Gt_data, dtype=np.uint8)
    #print(Gt_data.shape)
    gt_points = np.array(list(np.where(Gt_data))).transpose((1, 0))
    
    # for co in range(len(gt_points)):
    #     gt_x = gt_points[co][1]
    #     gt_y = gt_points[co][0]
    #     cv2.circle(Img_data, (int(gt_x), int(gt_y)), 2, (0,0,255), -1)
    # Img_data[Gt_data>0] = 255
    cv2.imwrite(save_vis_path + img_train[k], Img_data)
    #print(Gt_data.shape)
    #cell coordinate generate
    coordinate = np.column_stack((np.where(Gt_data)))
    print(len(coordinate))
    sumdot.append(len(coordinate))


    #save cell coordinate
    scipy.io.savemat(os.path.join(save_train_path, gt_train[k].split('.')[0]+'.mat'), mdict={'coordinate': coordinate})

sumdot = np.array(sumdot, dtype=np.int32)
print(sumdot.mean())