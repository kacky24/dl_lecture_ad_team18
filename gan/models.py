import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# variable, function names are similar to those in the article
class Generator(nn.Module):
    def __init__(self, inchannel_num, outchannel_num=3, K=64):
        super(Generator, self).__init__()
        self.cbp0 = CBP(inchannel_num, K, (3, 3), (1, 1), (1, 1))
        self.cbp1 = CBP(K, K, (3, 3), (1, 1), (1, 1))
        self.cbp2 = CBP(K, K, (3, 3), (1, 1), (1, 1))
        self.cbp3 = CBP(K, K, (3, 3), (1, 1), (1, 1))
        self.cbp4 = CBP(K, K // 2, (3, 3), (1, 1), (1, 1))
        self.cbp5 = CBP(K // 2, 1, (3, 3), (1, 1), (1, 1))
        self.dbr0 = DBR(1, K // 2, (3, 3), (1, 1), (1, 1))
        self.dbr1 = DBR(K // 2, K, (3, 3), (1, 1), (1, 1))
        self.dbr2 = DBR(K * 2, K, (3, 3), (1, 1), (1, 1))
        self.dbr3 = DBR(K, K, (3, 3), (1, 1), (1, 1))
        self.dbr4 = DBR(K * 2, K, (3, 3), (1, 1), (1, 1))
        self.dbr5 = DBR(K, outchannel_num, (3, 3), (1, 1), (1, 1))

    def forward(self, images):
        en0 = self.cbp0(images)
        en1 = self.cbp1(en0)
        en2 = self.cbp2(en1)
        en3 = self.cbp3(en2)
        en4 = self.cbp4(en3)
        en5 = self.cbp5(en4)
        de5 = self.dbr0(en5)
        de4 = self.dbr1(de5)
        # skip connection
        de3 = self.dbr2(torch.cat((en3, de4), 1))
        de2 = self.dbr3(de3)
        # skip connection
        de1 = self.dbr4(torch.cat((en1, de2), 1))
        de0 = self.dbr5(de1)
        # skip connection
        # out = F.tanh(torch.cat((images, de0), 1))
        out = F.tanh(de0)

        return out


class Discriminator(nn.Module):
    def __init__(self, inchannel_num, K=48):
        super(Discriminator, self).__init__()
        self.conv0 = nn.Conv2d(inchannel_num, K, (4, 4), (2, 2), (1, 1))
        self.prelu = nn.PReLU()
        # self.bn = nn.BatchNorm2d(K)
        self.cbp0 = CBP(K, 2*K, (4, 4), (2, 2), (1, 1))
        self.cbp1 = CBP(2*K, 4*K, (4, 4), (2, 2), (1, 1))
        self.cbp2 = CBP(4*K, 8*K, (4, 4), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(8*K, 1, (4, 4), (1, 1), (1, 1))

    def forward(self, images1, images2):
        '''
        images1: snow images
        images2: normal images or generated images
        '''
        images = torch.cat((images1, images2), 1)
        features = self.conv0(images)
        features = self.prelu(features)
        features = self.cbp0(features)
        features = self.cbp1(features)
        features = self.cbp2(features)
        features = F.sigmoid(self.conv1(features))

        return features


class CBP(nn.Module):
    '''
    convolution - batchnormalization - prelu
    '''
    def __init__(self, inchannel_num, outchannel_num, k_size, stride, padding):
        super(CBP, self).__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(inchannel_num, outchannel_num,
                          k_size, stride, padding),
                nn.BatchNorm2d(outchannel_num),
                nn.PReLU()
                )

    def forward(self, features):
        features = self.layers(features)
        return features


class DBR(nn.Module):
    '''
    deconvolution - batchnormalization - relu
    '''
    def __init__(self, inchannel_num, outchannel_num, k_size, stride, padding):
        super(DBR, self).__init__()
        self.layers = nn.Sequential(
                nn.ConvTranspose2d(inchannel_num, outchannel_num,
                                   k_size, stride, padding),
                nn.BatchNorm2d(outchannel_num),
                nn.ReLU()
                )

    def forward(self, features):
        features = self.layers(features)
        return features


class VggTransformar(nn.Module):
    '''
    vgg module for perceptual loss
    '''
    def __init__(self):
        super(VggTransformar, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        modules = list(vgg16.children())[0]
        modules = list(modules)[:9]
        self.vgg16 = nn.Sequential(*modules)

    def forward(self, images):
        features = self.vgg16(images)
        # features = self.vgg16(torch.cat((t_images, g_images), 1))
        return features
