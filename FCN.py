import os.path as osp

#import fcn
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
#from .fcn32s import get_upsampling_weight

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN16s(nn.Module):

    # pretrained_model = \
    #     osp.expanduser('~/data/models/pytorch/fcn16s_from_caffe.pth')

    # @classmethod
    # def download(cls):
    #     return fcn.data.cached_download(
    #         url='http://drive.google.com/uc?id=0B9P1L--7Wd2vVGE3TkRMbWlNRms',
    #         path=cls.pretrained_model,
    #         md5='991ea45d30d632a01e5ec48002cac617',
    #     )

    def __init__(self, n_class=2):
        super(FCN16s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.fc6_bn = nn.BatchNorm2d(4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, 32, stride=16, bias=False)

        # self._initialize_weights()

    # def _initialize_weights(net, init_type='normal', gain=0.02):
    #     def init_func(m):
    #     classname = m.__class__.__name__
    #     if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
    #         if init_type == 'normal':
    #             init.normal_(m.weight.data, 0.0, gain)
    #         elif init_type == 'xavier':
    #             init.xavier_normal_(m.weight.data, gain=gain)
    #         elif init_type == 'kaiming':
    #             init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #         elif init_type == 'orthogonal':
    #             init.orthogonal_(m.weight.data, gain=gain)
    #         else:
    #             raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             init.constant_(m.bias.data, 0.0)
    #     elif classname.find('BatchNorm2d') != -1:
    #         init.normal_(m.weight.data, 1.0, gain)
    #         init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    # net.apply(init_func)


    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1_bn(self.conv1_1(h)))
        h = self.relu1_2(self.conv1_2_bn(self.conv1_2(h)))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        h = self.pool5(h)

        h = self.relu6(self.fc6_bn(self.fc6(h)))
        h = self.drop6(h)

        h = self.relu7(self.fc7_bn(self.fc7(h)))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c

        h = self.upscore16(h)
        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
        # print('shape:',h.shape)

        return h

    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)