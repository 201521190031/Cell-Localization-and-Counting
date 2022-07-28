import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
from torchvision import transforms
import  time
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4,drop_last=False):
        if train:
            # root =4*root
            #random.shuffle(root)
            self.batch_size = batch_size
        else :
            self.batch_size = 1

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.drop_last = drop_last
        self.num_workers = num_workers



    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        fname = os.path.basename(img_path)
        imgs, target, sigma_map, k = load_data(img_path,self.train)

        for i, img in enumerate(imgs):
            if self.transform is not None:
                img = self.transform(img)
            imgs[i] = img

        imgs = torch.ToTensor(imgs)

        return imgs, target, fname, sigma_map, k



class listDataset_second(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        # if train:
        #     root =4*root
        # random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        fname = self.lines[index]['fname']
        feature = self.lines[index]['feature_1']
        gt_num = self.lines[index]['gt_num_1']

        return fname, feature, gt_num


class listDataset_ucf_1920(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        #if train:
            #root =4*root
            #random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]
        fname = os.path.basename(img_path)
        img,target,kpoint,sigma_map = load_data_ucf_1920(img_path,self.train)



        loader = transforms.ToTensor()
        original_img = loader(img.copy())


        if self.transform is not None:
            img = self.transform(img).cuda()


        target = target.copy()
        kpoint = kpoint.copy()
        sigma_map =sigma_map.copy()
        target = torch.from_numpy(target).cuda()

        crop_size_x = img.shape[1]/2
        crop_size_y = img.shape[2]/2

        x0 = 0
        y0 = 0
        img_return = img[:, x0 : x0 + crop_size_x, y0 : y0 + crop_size_y].unsqueeze(0)
        target_return = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)


        x0 = 0 + crop_size_x
        y0 = 0
        img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        img_return = torch.cat([img_return, img_crop],0)
        target_crop = target[x0 : x0 + crop_size_x, y0 : y0 + crop_size_y].unsqueeze(0)
        target_return = torch.cat([target_return, target_crop], 0)

        x0 = 0
        y0 = 0 + crop_size_y
        img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        img_return = torch.cat([img_return, img_crop],0)
        target_crop = target[x0 : x0 + crop_size_x, y0 : y0 + crop_size_y].unsqueeze(0)
        target_return = torch.cat([target_return, target_crop], 0)

        x0 = crop_size_x
        y0 = crop_size_y
        img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        img_return = torch.cat([img_return, img_crop],0)
        target_crop = target[x0 : x0 + crop_size_x, y0 : y0 + crop_size_y].unsqueeze(0)
        target_return = torch.cat([target_return, target_crop], 0)


        return img_return,target_return,fname,kpoint,sigma_map


class listDataset_ucf_1024(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=8):
        #if train:
            #root =4*root
            #random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        start = time.time()
        img_path = self.lines[index]
        fname = os.path.basename(img_path)
        img,target,k = load_data_1024(img_path,self.train)

        loader = transforms.ToTensor()
        original_img = loader(img.copy())


        if self.transform is not None:
            img = self.transform(img).cuda()

        end = time.time()

        return img,target,fname,k
        

class listDataset_Cell(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        # if train:
        #     root =4*root
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        gt_path = self.lines[index]
        fname = os.path.basename(gt_path)

        img, target, k, mask, img_raw = load_data_Cell(gt_path, self.train)
        # print(len(images))
        # imgs = torch.zeros((4, 3, images[0].size[0], images[0].size[1]))
        # for i, img in enumerate(images):
        #     img_raw = np.asarray(img.copy())
        if self.transform is not None:
            img = self.transform(img)
        #     imgs[i] = img
        #     # print(img.shape)
        # img_raws = np.array(img_raws)

        # img_raw = torch.from_numpy(img_raw)
        target = torch.from_numpy(target)

        return img, target, k, fname, mask, img_raw

class listDataset_Cell_Counter(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        # if train:
        #     root =4*root
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        gt_path = self.lines[index]
        fname = os.path.basename(gt_path)

        target, k = load_data_Cell_counter(gt_path, self.train)
        target = target.transpose((1,2,0))
        # print('pre_target:',target.shape)

        if self.transform is not None:
            target = self.transform(target)
        # print('target:',target.shape)
        
            # cls_l[i] = torch.from_numpy(cls_l[i])
        # imgs = torch.Tensor(imgs)

        return target, k


class listDataset_Cell_Val(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        # if train:
        #     root =4*root
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        gt_path = self.lines[index]
        fname = os.path.basename(gt_path)

        img, target, k, img_raw = load_data_Cell_val(gt_path, self.train)
        # imgs = torch.zeros((4, 3, images[0].size[0], images[0].size[1]))

        # for i, img in enumerate(images):
        if self.transform is not None:
            img = self.transform(img)
        #     imgs[i] = img
        # img_raws = np.array(img_raws)

        # img_raw = torch.from_numpy(img_raw)
        target = torch.from_numpy(target)
        # print(img.shape)

        return img, target, k, fname, img_raw



class listDataset_trancos_extract (Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]
        fname = os.path.basename(img_path)

        img, target, kpoint, sigma_map, mask = load_data_trancos_extract(img_path, self.train)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, kpoint, fname, sigma_map, mask


class listDataset_rate(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1, num_workers=4,
                 drop_last=False):
        if train:
            # root =4*root
            #random.shuffle(root)
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.drop_last = drop_last
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        #print(index,self.lines)

        img_path = self.lines[index]
        fname = os.path.basename(img_path)

        feature = np.asarray(h5py.File(img_path)['feature'])
        gt_num = np.asarray(h5py.File(img_path)['gt_num'])

        return  fname, feature, gt_num


class listDataset_faster(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1, num_workers=4,
                 drop_last=False):
        if train:
            # root =4*root
            random.shuffle(root)
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.drop_last = drop_last
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        start = time.time()
        img =self.lines[index]['img']
        target =self.lines[index]['gt']
        fname =self.lines[index]['fname']
        kpoint = self.lines[index]['kpoint']


        '''data augmention'''
        if self.train==True:
            if random.random() > 0.7:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.fliplr(kpoint)

            if random.random() > 0.7:
                proportion = random.uniform(0.004, 0.015)
                width, height = img.size[0], img.size[1]
                num = int(height * width * proportion)
                for i in range(num):
                    w = random.randint(0, width - 1)
                    h = random.randint(0, height - 1)
                    if random.randint(0, 1) == 0:
                        img.putpixel((w, h), (0, 0, 0))
                    else:
                        img.putpixel((w, h), (255, 255, 255))
            #print("train")

        target = target.copy()
        kpoint = kpoint.copy()
        img = img.copy()

        if self.transform is not None:
            img = self.transform(img)


        target = torch.from_numpy(target).cuda()

        crop_size_x = img.shape[1] / 2
        crop_size_y = img.shape[2] / 2

        x0 = 0
        y0 = 0
        img_return = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        target_return = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)

        x0 = 0 + crop_size_x
        y0 = 0
        img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        img_return = torch.cat([img_return, img_crop], 0)
        target_crop = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        target_return = torch.cat([target_return, target_crop], 0)

        x0 = 0
        y0 = 0 + crop_size_y
        img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        img_return = torch.cat([img_return, img_crop], 0)
        target_crop = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        target_return = torch.cat([target_return, target_crop], 0)

        x0 = crop_size_x
        y0 = crop_size_y
        img_crop = img[:, x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        img_return = torch.cat([img_return, img_crop], 0)
        target_crop = target[x0: x0 + crop_size_x, y0: y0 + crop_size_y].unsqueeze(0)
        target_return = torch.cat([target_return, target_crop], 0)
        end = time.time()
        #print("load time",end-start)
        return img_return, target_return, fname,kpoint



class listDataset_25(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1, num_workers=4,
                 drop_last=False):
        if train:
            # root =4*root
            #random.shuffle(root)
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.drop_last = drop_last
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        start = time.time()
        img_path = self.lines[index]

        fname = os.path.basename(img_path)
        img,target,kpoint,sigma_map= load_data_25(img_path,self.train)


        target = target.copy()
        kpoint = kpoint.copy()
        img = img.copy()
        #sigma_map = sigma_map.copy()
        if self.transform is not None:
            img = self.transform(img)

        img_return = img
        target_return = target
        
        end = time.time()
        #print("load time",end-start)
        return img_return, target_return, fname,kpoint,sigma_map

class listDataset_B_25(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4,
                 drop_last=False):
        if train:
            # root =4*root
            # random.shuffle(root)
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.drop_last = drop_last
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        start = time.time()
        img_path = self.lines[index]

        fname = os.path.basename(img_path)
        img, target, kpoint, sigma_map = load_data_B_25(img_path, self.train)

        target = target.copy()
        kpoint = kpoint.copy()
        img = img.copy()
        # sigma_map = sigma_map.copy()
        if self.transform is not None:
            img = self.transform(img)

        img_return = img
        target_return = target

        return img_return, target_return, fname, kpoint, sigma_map