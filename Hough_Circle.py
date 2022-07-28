import cv2
import numpy as np
import h5py
import os
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from utils import caculate_precision_baseline, caculate_recall_baseline
from nms import *

with open('./Cellstest.npy', 'rb') as outfile:
    val_list = np.load(outfile).tolist()
print(val_list)

precisions = []
recalls = []
maes = []
rads = []

# dir to save result
s_dir = 'CHT/CHTResult(thresh=30,31)/'
if not os.path.exists(s_dir):
    os.mkdir(s_dir)


for name in val_list:
    
    # Load the image and transform it to grayscale
    img = cv2.imread(name.replace('dots','cell').replace('.h5','.png'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # print(gray.max())

    # get GT locations
    gtpoint = np.asarray(h5py.File(name, 'r')['kpoint'])
    gt = np.argwhere(gtpoint)
    gt_count = gtpoint.sum()

    #get edges
    edges = canny(gray, sigma=3, low_threshold=30, high_threshold=31)

    # Detect two recalldii
    hough_radii = np.arange(3, 5, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circles
    circles = []
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=200)

    circles = [cx, cy, radii]
    circles = list(map(list, zip(*circles)))
    circles.sort(key=lambda x: (x[0], x[1]))

    min_index = nms(circles, 2, img.shape)
    min_index = np.unique(min_index)
    for i, index in enumerate(min_index):
        del circles[index-i]

    print('Circle:', len(circles))

    pre_count = len(circles)
    mae = abs(pre_count - gt_count)
    maes.append(mae)
    rad = np.mean(radii)
    rads.append(rad)

    #get fname to save result
    fname = name.split('/')[-1].replace('.h5', '.bmp')

    #save edge
    edges = np.uint8(edges)
    edges = 255.0 * (edges - edges.min())/ (edges.max() - edges.min())
    cv2.imwrite(s_dir+fname.replace('.bmp', 'Edge.bmp'), edges)

    # jduge where circles are empty
    if len(circles) == 0:
        cv2.imwrite(s_dir+fname, gray)
        continue

    #Get precision and recall
    precision = caculate_precision_baseline(img, circles, gt, (256, 256), fname, s_dir)
    recall = caculate_recall_baseline(gray, circles, gt, (256, 256), fname)
    precisions.append(precision)
    recalls.append(recall)

precisions = np.array(precisions, dtype=np.float32)*100.0
recalls = np.array(recalls, dtype=np.float32)*100.0
maes = np.array(maes, dtype=np.float32)
rads = np.array(rads, dtype=np.float32)
recall = recalls.mean()
precision = precisions.mean()
mae = maes.mean()
rad = rads.mean()

#write it to the log
with open('logs/CHT/hough00(thresh=30,31).txt', 'a+') as f:
    f.write('Epoch:{epoch:d} * MAE {mae:.3f} , * Precision {precision:.3f} ,* Recall {recall:.3f}\n* All precision {precisions:},  * All recall {recalls:}, * Radius {radius:}\n'
        .format(epoch=1, mae=mae, precision=precision, recall=recall, precisions= precisions, recalls= recalls, radius=rad))
    print('Epoch:{epoch:d} * MAE {mae:.3f} , * Precision {precision:.3f} ,* Recall {recall:.3f}\n* All precision {precisions:},  * All recall {recalls:}, * Radius {radius:}'
        .format(epoch=1, mae=mae, precision=precision, recall=recall, precisions= precisions, recalls= recalls, radius=rad))

