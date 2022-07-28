import numpy as np
import cv2
import math
import scipy.io
import os
import h5py
import scipy.misc

# img_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/'
# gt_train_path = './part_A_final/train_data/ground_truth/'
# save_train_path = './part_A_final_25/train_data/'


img_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/images/'
gt_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth/'
save_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final_25'
if not os.path.exists(save_train_path):
    os.mkdir(save_train_path)
    os.mkdir(os.path.join(save_train_path, 'train_data'))
    save_train_path = os.path.join(save_train_path, 'train_data')
# if not os.path.exists(save_path_A_test):
#     os.mkdir(save_path_A_test)


# img_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data/images/'
# gt_train_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data/ground_truth/'
# img_test_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/images/'
# gt_test_path = '../datasets/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/ground_truth/'



distance = 1

def Distance_generate(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda*size[1], lamda*size[0]), 0)

    new_size = new_im_data.shape
    #print(new_size[0], new_size[1])
    # d_map = np.zeros((new_size[0],new_size[0][1])).astype(np.uint8)
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda*gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y]-255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 5)

    distance_map[(distance_map >= 0) & (distance_map < 1 * distance)] = 0
    distance_map[(distance_map >= 1 * distance) & (distance_map < 2 * distance)] = 1
    distance_map[(distance_map >= 2 * distance) & (distance_map < 3 * distance)] = 2
    distance_map[(distance_map >= 3 * distance) & (distance_map < 4 * distance)] = 3
    distance_map[(distance_map >= 4 * distance) & (distance_map < 5 * distance)] = 4
    distance_map[(distance_map >= 5 * distance) & (distance_map < 6 * distance)] = 5
    distance_map[(distance_map >= 6 * distance) & (distance_map < 8 * distance)] = 6
    distance_map[(distance_map >= 8 * distance) & (distance_map < 12 * distance)] = 7
    distance_map[(distance_map >= 12 * distance) & (distance_map < 18 * distance)] = 8
    distance_map[(distance_map >= 18 * distance) & (distance_map < 28 * distance)] = 9
    distance_map[(distance_map >= 28 * distance)] = 10
    # distance_map[(distance_map >= 0) & (distance_map < 1 * distance)] = 0
    # distance_map[(distance_map >= 1 * distance) & (distance_map < 2 * distance)] = 1
    # distance_map[(distance_map >= 2 * distance) & (distance_map < 3 * distance)] = 2
    # distance_map[(distance_map >= 3 * distance) & (distance_map < 4 * distance)] = 3
    # distance_map[(distance_map >= 4 * distance) & (distance_map < 6 * distance)] = 4
    # distance_map[(distance_map >= 6 * distance) & (distance_map < 12 * distance)] = 5
    # distance_map[(distance_map >= 12 * distance) & (distance_map < 20 * distance)] = 6
    # distance_map[(distance_map >= 20 * distance) & (distance_map < 28 * distance)] = 7
    # distance_map[(distance_map >= 28 * distance)] = 8


    return new_im_data, distance_map


img_train = []
gt_train = []

for file_name in os.listdir(img_train_path):
    if file_name.split('.')[1] == 'jpg':
        img_train.append(file_name)

for file_name in os.listdir(gt_train_path):
    if file_name.split('.')[1] == 'mat':
        gt_train.append(file_name)

img_train.sort()
gt_train.sort()



for k in range(len(img_train)):

    Img_data = cv2.imread(img_train_path + img_train[k])
    Gt_data = scipy.io.loadmat(gt_train_path + gt_train[k])
    print(Gt_data)


    Gt_data = Gt_data['image_info'][0][0][0][0][0]
    result = Distance_generate(Img_data, Gt_data, 1)
    new_img = result[0]
    Distance_map = result[1]

    patch_x = Img_data.shape[1] / 2
    patch_y = Img_data.shape[0] / 2
    Gt_data_ori = Gt_data.copy()
    if ((Img_data.shape[0] > 850 and Img_data.shape[1] > 400) or (Img_data.shape[0] > 400 and Img_data.shape[1] > 850))  and len(Gt_data_ori)>30000:
        for i in range(2):
            for j in range(2):
                gt_data = []
                new_img = Img_data[i * patch_y:(i + 1) * patch_y, j * patch_x:(j + 1) * patch_x]
                for n in range(len(Gt_data)):

                    if Gt_data_ori[n][0] >= j * patch_x and Gt_data_ori[n][0] < (j + 1) * patch_x and Gt_data_ori[n][
                        1] >= i * patch_y and Gt_data_ori[n][1] < (i + 1) * patch_y:
                        Gt_data[n][0] = Gt_data_ori[n][0] - j * patch_x
                        Gt_data[n][1] = Gt_data_ori[n][1] - i * patch_y
                        gt_data.append(Gt_data[n])

                '''gengrate kpoint'''
                kpoint = np.zeros((new_img.shape[0], new_img.shape[1])).astype(np.uint8)
                for count in range(0, len(gt_data)):
                    if int(gt_data[count][1]) < new_img.shape[0] and int(gt_data[count][0]) < new_img.shape[1]:
                        kpoint[int(gt_data[count][1]), int(gt_data[count][0])] = 1

                result = Distance_generate(new_img, gt_data, 1)
                Distance_map = result[1]

                new_img_path = (save_train_path + img_train[k]).split('.jpg')[0] + "_" + str(i) + str(j) + '.jpg'
                # if len(gt_data) == 0:
                # 	print(new_img_path)
                # 	continue

                gt_show_path = new_img_path.split('.jpg')[0] + 'gt.jpg'
                h5_path = new_img_path.split('.jpg')[0] + '.h5'
                with h5py.File(h5_path, 'w') as hf:
                    hf['distance'] = Distance_map
                    hf['kpoint'] = kpoint


                print(k, new_img_path, h5_path)
                cv2.imwrite(new_img_path, new_img)
                Distance_map = Distance_map / np.max(Distance_map) * 255
                cv2.imwrite(gt_show_path, Distance_map)

    else:
        print(save_train_path)

        result = Distance_generate(Img_data, Gt_data, 1)
        Distance_map = result[1]


        new_img_path = os.path.join(save_train_path, img_train[k])

        mat_path = new_img_path.split('.jpg')[0]
        gt_show_path = new_img_path.split('.jpg')[0] + 'gt.jpg'
        h5_path = new_img_path.split('.jpg')[0] + '.h5'

        '''gengrate kpoint'''
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1])).astype(np.uint8)
        for count in range(0, len(Gt_data)):
            if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

        '''generate sigma'''
        pts = np.array(np.column_stack((np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
        leafsize = 2048
        # build kdtree
        #print(np.shape(pts))

        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=8)
        sigma_map = np.zeros(kpoint.shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            sigma = (distances[i][1]) / 2
            sigma_map[pt[1], pt[0]] = sigma
        '''' end '''

        print(kpoint.sum())
        with h5py.File(h5_path, 'w') as hf:
            hf['distance'] = Distance_map
            hf['kpoint'] = kpoint
            hf['sigma_map'] = sigma_map


        # print(k,h5_path, new_img_path)
        #
        # cv2.imwrite(new_img_path, Img_data)
        # Distance_map = Distance_map / np.max(Distance_map) * 255
        # cv2.imwrite(gt_show_path, Distance_map)

    cv2.waitKey(1)


