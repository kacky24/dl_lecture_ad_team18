import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, inchannel_num, outchannel_num):
        super(Generator, self).__init__()
        # encoder
        self.conv = nn.Conv2d(inchannel_num, 64, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.cbl0 = CBL(64, 128, 4, 2, 1)
        self.cbl1 = CBL(128, 256, 4, 2, 1)
        self.cbl2 = CBL(256, 512, 4, 2, 1)
        self.cbl3 = CBL(512, 512, 4, 2, 1)
        self.cbl4 = CBL(512, 512, 4, 2, 1)
        self.cbl5 = CBL(512, 512, 4, 2, 1)
        self.cbl6 = CBL(512, 512, 4, 2, 1)
        # decoder
        self.dbdr0 = DBDR(512, 512, 4, 2, 1)
        self.dbdr1 = DBDR(1024, 512, 4, 2, 1)
        self.dbdr2 = DBDR(1024, 512, 4, 2, 1)
        self.dbr0 = DBR(1024, 512, 4, 2, 1)
        self.dbr1 = DBR(1024, 256, 4, 2, 1)
        self.dbr2 = DBR(512, 128, 4, 2, 1)
        self.dbr3 = DBR(256, 64, 4, 2, 1)
        self.dconv = nn.ConvTranspose2d(128, outchannel_num, 4, 2, 1)

    def forward(self, images):
        # encoder
        en0 = self.lrelu(self.conv(images))
        en1 = self.cbl0(en0)
        en2 = self.cbl1(en1)
        en3 = self.cbl2(en2)
        en4 = self.cbl3(en3)
        en5 = self.cbl4(en4)
        en6 = self.cbl5(en5)
        en7 = self.cbl6(en6)
        # decoder
        de7 = self.dbdr0(en7)
        de6 = self.dbdr1(torch.cat((de7, en6), 1))
        de5 = self.dbdr2(torch.cat((de6, en5), 1))
        de4 = self.dbr0(torch.cat((de5, en4), 1))
        de3 = self.dbr1(torch.cat((de4, en3), 1))
        de2 = self.dbr2(torch.cat((de3, en2), 1))
        de1 = self.dbr3(torch.cat((de2, en1), 1))
        de0 = self.dconv(torch.cat((de1, en0), 1))

        return F.tanh(de0)


class Discriminator70(nn.Module):
    def __init__(self, inchannel_num):
        super(Discriminator70, self).__init__()
        self.conv0 = nn.Conv2d(inchannel_num, 64, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.cbl0 = CBL(64, 128, 4, 2, 1)
        self.cbl1 = CBL(128, 256, 4, 2, 1)
        self.cbl2 = CBL(256, 512, 4, 2, 1)
        self.conv1 = nn.Conv2d(512, 1, 4, 2, 1)

    def forward(self, images1, images2):
        images = torch.cat((images1, images2), 1)
        features = self.conv0(images)
        features = self.lrelu(features)
        features = self.cbl0(features)
        features = self.cbl1(features)
        features = self.cbl2(features)
        features = F.sigmoid(self.conv1(features))

        return features


class Discriminator16(nn.Module):
    def __init__(self, inchannel_num):
        super(Discriminator16, self).__init__()
        self.conv0 = nn.Conv2d(inchannel_num, 64, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.cbl = CBL(64, 128, 4, 2, 1)
        self.conv1 = nn.Conv2d(128, 1, 4, 2, 1)

    def forward(self, images1, images2):
        images = torch.cat((images1, images2), 1)
        features = self.conv0(images)
        features = self.lrelu(features)
        features = self.cbl(features)
        features = self.conv1(features)

        return F.sigmoid(features)


class CBL(nn.Module):
    '''
    convolution - batchnormalization - leaky_relu
    '''
    def __init__(self, inchannel_num, outchannel_num, k_size, stride, padding):
        super(CBL, self).__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(inchannel_num, outchannel_num,
                          k_size, stride, padding),
                nn.BatchNorm2d(outchannel_num),
                nn.LeakyReLU(negative_slope=0.2)
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


class DBDR(nn.Module):
    '''
    deconvolution - batchnormalization - dropout- relu
    '''
    def __init__(self, inchannel_num, outchannel_num, k_size, stride, padding):
        super(DBDR, self).__init__()
        self.layers = nn.Sequential(
                nn.ConvTranspose2d(inchannel_num, outchannel_num,
                                   k_size, stride, padding),
                nn.BatchNorm2d(outchannel_num),
                nn.Dropout(),
                nn.ReLU()
                )

    def forward(self, features):
        features = self.layers(features)
        return features
