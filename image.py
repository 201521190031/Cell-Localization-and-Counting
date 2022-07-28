import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import scipy
import time
import jpeg4py as jpeg4
import torch
from utils import get_flux

def load_data(img_path, train=True):
    #print(img_path)
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    print(img_path)
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['distance'])
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    gt = mat["image_info"][0, 0][0, 0][0]
    k = np.zeros((img.size[1], img.size[0]))

    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.size[1] and int(gt[i][0]) < img.size[0]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    pts = np.array(zip(np.nonzero(k)[1], np.nonzero(k)[0]))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=2)
    sigma_map = np.zeros(k.shape, dtype=np.float32)
    # pt2d = np.zeros(k.shape,dtype= np.float32)
    for i, pt in enumerate(pts):
        sigma = (distances[i][1] )/2

        sigma_map[pt[1], pt[0]] = sigma

    if train==True:

        if random.random() > 0.6:
            proportion = random.uniform(0.001, 0.01)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))
            #print("noise")

        img=img.copy()
        target=target.copy()
        sigma_map = sigma_map.copy()
        k = k.copy()


    return img, target,k,sigma_map


def load_data_ucf_1920(img_path, train=True):

    img_path = img_path.decode()

    img_path = img_path.replace('.h5', '.jpg')
    #print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['distance'])
    kpoint = np.asarray(gt_file['kpoint'])
    sigma_map = np.asarray(gt_file['sigma_map'])

    if train==True:

        width, height = int(img.size[0]*0.7), int(img.size[1]*0.7)
        #print(img.size, target.shape,kpoint.shape, sigma_map.shape,width,height)
        start_x = random.randint(0,img.size[0]-width)
        start_y = random.randint(0,img.size[1]-height)
        img = img.crop((start_x, start_y, width+start_x, height+start_y))
        target = target[start_y : start_y + height, start_x : start_x+width]
        kpoint = kpoint[start_y : start_y + height, start_x : start_x+width]
        sigma_map = sigma_map[start_y : start_y + height, start_x : start_x+width]

    img = img.copy()
    target = target.copy()
    kpoint = kpoint.copy()
    sigma_map = sigma_map.copy()

    img.save('1_ucf.jpg')

    return img, target, kpoint,sigma_map

def load_data_1024(img_path, train=True):
    start = time.time()
    img_path = img_path.decode()

    img_path = img_path.replace('.h5', '.jpg')
    #print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path).convert('RGB')

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['distance'])
    k = np.asarray(gt_file['kpoint'])
    k_path = img_path.replace('.jpg', '.npy')
    gt = np.load(k_path)

    end = time.time()

    img = img.copy()
    target=target.copy()

    k = k.copy()

    #print(img.size, target.shape, k.sum(), len(gt))
    img.save('2_ucf.jpg')

    return img, target, k

def load_data_Cell(gt_path, train=True):

    #Cells
    img_path = gt_path.replace('dots','cell').replace('.h5','.png')

    img = Image.open(img_path).convert('RGB')
    img_raw = cv2.imread(img_path)
    gt_file = h5py.File(gt_path)


    target = np.asarray(gt_file['flux'])
    # sigma_map = np.asarray(gt_file['sigma_map'])
    k = np.asarray(gt_file['kpoint'])
    mask = np.asarray(gt_file['mask'])

    if train == True:
        
        if random.random() > 0.5:
            k = np.fliplr(k)
            img_raw = np.flip(img_raw, 1).copy()
            # print('img_raw:',img_raw.shape)
            target, mask = get_flux(k)

            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            proportion = random.uniform(0.001, 0.015)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))

    target = target.copy()

    k = k.copy()

    return img, target, k, mask, img_raw
    

def load_data_Cell_counter(gt_path, train=True):
    
    
    gt_file = h5py.File(gt_path)

    target = np.asarray(gt_file['flux'])
    img_path = gt_path.replace('dots','cell').replace('.h5','.png')
    img = np.asarray(cv2.imread(img_path))
    img = img.transpose(2,1,0).transpose(0,2,1)

    k = np.asarray(gt_file['kpoint'])
    target = np.concatenate((target, img), axis=0)

    
    if train == True:
        if random.random() > 0.5:
            target = np.fliplr(target)
            k = np.fliplr(k)
            # print("noise")
    target = target.copy()

    k = k.copy()
    k = k.sum()

    return target, k


def load_data_Cell_val(gt_path, train=False):

    img_path = gt_path.replace('dots','cell').replace('.h5','.png')

    img = Image.open(img_path).convert('RGB')
    # img = Image.open(img_path)
    img_raw = cv2.imread(img_path)

    gt_file = h5py.File(gt_path)

    target = np.asarray(gt_file['flux'])
    # sigma_map = np.asarray(gt_file['sigma_map'])
    k = np.asarray(gt_file['kpoint'])
    if train == True:

        if random.random() > 0.5:
            proportion = random.uniform(0.001, 0.015)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))

    # img = img.copy()
    target = target.copy()

    k = k.copy()

    return img, target, k, img_raw

def load_data_trancos_extract(img_path, train=True):

    gt_path = img_path.replace('.jpg', '.h5')
    img = Image.open(img_path.replace('.h5','.jpg')).convert('RGB')
    gt_file = h5py.File(gt_path)

    target = np.asarray(gt_file['distance'])
    sigma_map = np.asarray(gt_file['sigma_map'])
    kpoint =  np.asarray(gt_file['kpoint'])

    img = img.copy()
    target = target.copy()
    sigma_map = sigma_map.copy()
    #mask = mask.copy()

        #k = k.copy()
    kpoint = kpoint.copy()

    return img, target, kpoint, sigma_map


def load_data_25(img_path, train=True):
    # print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    gt_file = h5py.File(gt_path)
    img_path = img_path.replace('h5','jpg').replace('part_A_final_25','part_A_final').replace('test_data/','test_data/images/').replace('train_data/','train_data/images/')
    print(img_path)
    img_path = gt_path.replace('h5','jpg')
    img = Image.open(img_path).convert('RGB')

    # print(gt_path)
    target = np.asarray(gt_file['distance'])
    k =  np.asarray(gt_file['kpoint'])
    sigma_map =  np.asarray(gt_file['sigma_map'])
    if train==True:


        if random.random() > 0.7:
            proportion = random.uniform(0.001, 0.01)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))
            #print("noise")

        img=img.copy()
        target=target.copy()
        k = k.copy()
        sigma_map = sigma_map.copy()
    # sigma_map = [1]
    img.save('1.jpg')


    return img, target,k,sigma_map


def load_data_B_25(img_path, train=True):
    # print(img_path)
    gt_path = img_path.replace('.jpg', '.h5')
    gt_file = h5py.File(gt_path)
    img_path = img_path.replace('h5','jpg').replace('part_A_final_25','part_A_final').replace('part_B_final_25','part_B_final').replace('test_data/','test_data/images/').replace('train_data/','train_data/images/')
    #print(img_path)
    img = Image.open(img_path).convert('RGB')

    # print(gt_path)
    target = np.asarray(gt_file['distance'])
    k =  np.asarray(gt_file['kpoint'])
    sigma_map =  np.asarray(gt_file['sigma_map'])
    if train==True:

        if random.random() > 0.6:
            target = np.fliplr(target)
            k = np.fliplr(k)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            sigma_map = np.fliplr(sigma_map)


        if random.random() > 0.7:
            proportion = random.uniform(0.001, 0.01)
            width, height = img.size[0], img.size[1]
            num = int(height * width * proportion)
            for i in range(num):
                w = random.randint(0, width - 1)
                h = random.randint(0, height - 1)
                if random.randint(0, 1) == 0:
                    img.putpixel((w, h), (0, 0, 0))
                else:
                    img.putpixel((w, h), (255, 255, 255))
            #print("noise")

        img=img.copy()
        target=target.copy()
        k = k.copy()
        sigma_map = sigma_map.copy()
    # sigma_map = [1]
    img.save('1.jpg')
   

    return img, target,k,sigma_map
#
