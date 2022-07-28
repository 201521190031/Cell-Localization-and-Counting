'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.init as init


cfg = {
    'VGG11': [16, 'M', 32, 'M', 64, 'M',128, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



class Counter(nn.Module):
    def __init__(self, vgg_name):
        super(Counter, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linear0 = nn.Sequential(nn.Linear(128*16*16, 200), nn.ReLU(inplace=True))
        self.linear1 = nn.Sequential(nn.Linear(200, 1), nn.ReLU(inplace=True))
        # self.linear2 = nn.Linear(10, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        out = torch.flatten(out, 1)
        # out = out.reshape((out.size(0), out.size(2), out.size(3), out.size(1)))
        out = self.linear0(out)
        out = self.linear1(out)
        # out = self.linear2(out)
        # out = out.reshape((out.size(0), out.size(3), out.size(1), out.size(2)))
        # print(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 5
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        # layers += [nn.Conv2d(x, 64, kernel_size=1), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


class Counter_Loss(nn.Module):
    """docstring for Counter_Loss"""
    def __init__(self):
        super(Counter_Loss, self).__init__()


    def forward(self,predict,target):
        # print('target:',target.shape)
        # print('predict:',predict.shape)
        means=torch.mean(torch.abs(predict-target))#torch.size([])
    
        countloss=1 - 1/(1+means)#torch.size([])
        return countloss



def test():
    net = Counter('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    test()

# test()