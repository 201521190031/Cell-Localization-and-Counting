import cv2
import numpy as np
import os
import glob
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
import h5py
import math
from PIL import Image,ImageDraw

def radExtract(sk):
    rads = np.zeros(sk.shape)
    locs = np.argwhere(sk>0)
    print('locs:',locs.shape)
    for o, loc in enumerate(locs):
        center_x = min(int(loc[1]), sk.shape[1]-1)
        center_y = min(int(loc[0]), sk.shape[1]-1)
        min_distance = 1e8
        

        for j in range(len(locs)):
            if j == o:
                continue
            cal_x = locs[j][1]
            cal_y = locs[j][0]

            
            distance = math.sqrt((float(center_x - cal_x)) * (center_x - cal_x) + (float(center_y - cal_y)) * (center_y - cal_y))

            if distance < min_distance:
                min_distance = distance
                # min_index = j

        # min_distance = min(min_distance, 10)
        # for counternet
        # min_distance = 12
        min_distance = min(min_distance, 12)
        rad = max(1, int(0.5*min_distance))
        rads[center_y][center_x] = rad
    return rads


def skel2seg(sk, rad, thresh=0.5, ratio=1):
    '''
    given skeleton and diameter, recover segmentation
    Args:
        sk:    the skeleton map, can be binray or probability map
        rad:   the radius map, should be of the same size as `sk`
        ratio: the ratio multiplied
    Returns:
        an PIL image with drawn circles
    '''
    sk = sk.copy()
    sk[sk < thresh] = 0
    sk[sk >= thresh] = 1
    img = Image.new('L', (sk.shape[1], sk.shape[0]))
    if ratio <= 0.01:
        return Image.fromarray(sk, 'L')
    draw = ImageDraw.Draw(img)
    index = np.argwhere(sk > 0)
    # print('Index:',index.shape)
    for i in range(len(index)):
        y = index[i][0]
        x = index[i][1]
        r = rad[y][x] * float(ratio)
        draw.ellipse([x - r, y - r, x +
                      r, y + r], fill=1)
        
    img = np.asarray(img, dtype=np.uint8)
    return img


def loadsklarge(imgidx, gtidx, save_path):
    # load image and skeleton
    image = cv2.imread(imgidx, 1)
    cls_l = np.zeros(4)
    new_size = (int(image.shape[0]), int(image.shape[1]))
    image = cv2.resize(image, new_size)
    skl = cv2.imread(gtidx, 0)
    skl = (skl > 0).astype(np.uint8)
    
    kpoints  = np.array(list(np.where(skl>0))).transpose((1,0))
    skeleton = np.zeros(new_size)
    for o in range(0, len(kpoints)):
        x = int(max(1, kpoints[o][0]))
        y = int(max(1, kpoints[o][1]))
        # print(len(gt),x,y)
        if x >= new_size[0] - 1 or y >= new_size[1] - 1:
            # print(o)
            continue
        skeleton[x][y] = 1
    # k_density = skeleton.copy()
    # k_density = gaussian_filter(k_density, 2)
    # print('k_density:',k_density.max())

    # normalization
    image = image.astype(np.float32)
    
    
    rads = radExtract(skeleton)
    dilmask = skel2seg(skeleton, rads)
    rev = 1-skeleton
    height = rev.shape[0]
    width = rev.shape[1]
    rev = (rev > 0).astype(np.uint8)
    dst, labels = cv2.distanceTransformWithLabels(rev, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
    index = np.copy(labels) 
    index[rev > 0] = 0
    place = np.argwhere(index > 0)

    nearCord = place[labels-1,:]
    x = nearCord[:, :, 0]
    y = nearCord[:, :, 1]
    nearPixel = np.zeros((2, height, width))
    nearPixel[0,:,:] = x
    nearPixel[1,:,:] = y
    grid = np.indices(rev.shape)
    grid = grid.astype(float)
    diff = grid - nearPixel

    dist = np.sqrt(np.sum(diff**2, axis = 0))

    direction = np.zeros((2, height, width), dtype=np.float32)
    direction[0,rev > 0] = np.divide(diff[0,rev > 0], dist[rev > 0])
    direction[1,rev > 0] = np.divide(diff[1,rev > 0], dist[rev > 0])

    direction[0] = direction[0]*(dilmask > 0)
    direction[1] = direction[1]*(dilmask > 0)

    flux = -1*np.stack((direction[0], direction[1]))
    norm = np.sqrt(direction[0]**2+direction[1]**2)
    norm = 255.0*norm/norm.max()

    dilmask = (dilmask>0).astype(np.float32)

    h5_path = os.path.join(save_path, gtidx.split('/')[-1].replace('.png', '.h5'))
    with h5py.File(h5_path, 'w') as hf:
        hf['flux'] = flux
        # hf['img'] = images
        hf['kpoint'] = skeleton
        hf['mask'] = dilmask
        # hf['large'] = enlarge
        # hf['class'] = cls_l
    print(dilmask.shape)

    cv2.imwrite(os.path.join(save_path, imgidx.split('/')[-1]), image)
    # cv2.imwrite(os.path.join(save_path, gtidx.split('/')[-1].replace('.png', '.jpg')), norm)

    return image, flux, dilmask

path = '../datasets/cells/'
save_path = '../datasets/cells/transxy/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


gtpaths = glob.glob(os.path.join(path, 'data/*dots.png'))
imgpaths = glob.glob(os.path.join(path, 'data/*cell.png'))
print(gtpaths)
gtpaths.sort()
imgpaths.sort()

for imgpath, gtpath in zip(imgpaths, gtpaths):
    img, flux, dialmask= loadsklarge(imgpath, gtpath, save_path)
    # loadsklarge(imgpath, gtpath, save_path)
    # flux = flux.transpose((1, 2, 0))
    # dialmask = dialmask.transpose((1, 2, 0))
    dialmask = 255.0*(dialmask/dialmask.max())
    # enlargek = 255.0*(enlargek/enlargek.max())
    # enlargemask = 255.0*(enlargemask/enlargemask.max())
    # print(flux.shape)
    cv2.imwrite(os.path.join(save_path, imgpath.split('/')[-1]), img)
    # cv2.imwrite(os.path.join(save_path, imgpath.split('/')[-1].replace('.png', 'Density.png')), density)
    cv2.imwrite(os.path.join(save_path, gtpath.split('/')[-1].replace('.png', 'Mask.png')), dialmask)
    # cv2.imwrite(os.path.join(save_path, gtpath.split('/')[-1].replace('.png', 'LargeImg.png')), enlarge)
    # cv2.imwrite(os.path.join(save_path, gtpath.split('/')[-1].replace('dots.png', 'flux.png')), flux)
