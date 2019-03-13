import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Input(nn.Module):
    def __init__(self, outChans):
        super(Input, self).__init__()
        self.conv = nn.Conv3d(1, outChans, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm3d(outChans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class SingleLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, dilation_rate=1):
        super(SingleLayer, self).__init__()
        self.bn = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, growth_rate, kernel_size=3,
                              dilation=dilation_rate, padding=dilation_rate, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        out = torch.cat((x, out), 1)
        return out


class Bottleneck(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, dilation_rate=1):
        super(Bottleneck, self).__init__()
        interChannels = 4*growth_rate
        self.bn1 = nn.BatchNorm3d(num_input_features)
        self.conv1 = nn.Conv3d(num_input_features, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growth_rate, kernel_size=3,
                               dilation=dilation_rate, padding=dilation_rate, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        out = torch.cat((x, out), 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, dilation_rate=1):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(int(num_layers)):
            layers.append(SingleLayer(num_input_features + i * growth_rate, growth_rate, drop_rate,
                                      dilation_rate=dilation_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, growth_rate, dilation_rate, drop_out):
        super(DenseAsppBlock, self).__init__()

        interChannels = 4 * growth_rate

        self.bn1 = nn.BatchNorm3d(input_num)
        self.conv1 = nn.Conv3d(input_num, interChannels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growth_rate, kernel_size=3, dilation=dilation_rate, padding=dilation_rate, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.drop_rate = drop_out

    def forward(self, x):

        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)

        out = torch.cat((x, out), 1)

        return out


class Trans(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(Trans, self).__init__()
        self.bn = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out


class DownTrans(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(DownTrans, self).__init__()
        self.bn = nn.BatchNorm3d(num_input_features)
        self.down_conv = nn.Conv3d(num_input_features, num_input_features, kernel_size=2, stride=2, bias=False)

        self.bn2 = nn.BatchNorm3d(num_input_features)
        self.conv2 = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.down_conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        out = self.conv2(self.relu(self.bn2(out)))
        return out


class UpTrans(nn.Module):
    def __init__(self, num_input_features, num_skip_features, num_output_features, drop_rate):
        super(UpTrans, self).__init__()

        self.bn1 = nn.BatchNorm3d(num_input_features)
        self.up_conv1 = nn.ConvTranspose3d(num_input_features, num_input_features, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm3d(num_input_features + num_skip_features)
        self.conv2 = nn.Conv3d(num_input_features + num_skip_features, num_output_features, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.drop_rate = drop_rate

    def forward(self, x, skipx):
        out = self.up_conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        out = torch.cat((out, skipx), 1)
        out = self.conv2(self.relu(self.bn2(out)))
        return out


class Output(nn.Module):
    def __init__(self, inChans):
        super(Output, self).__init__()
        self.bn1 = nn.BatchNorm3d(inChans)
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1, bias=False)
        self.softmax = F.softmax

    def forward(self, x):
        # convolve to 2 chan
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out, 1)
        # treat channel 0 as the predicted output
        return out


class Output2(nn.Module):
    def __init__(self, inChans):
        super(Output2, self).__init__()
#        self.conv = nn.Conv3d(2, 2, kernel_size=1, bias=False)
        self.softmax = F.softmax

    def forward(self, x):
        # convolve to 2 chan
#        out = self.conv(x)
        # make channels the last axis
        out = x.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out, 1)
        # treat channel 0 as the predicted output
        return out


class DenseNet_DenseASPP_FullVNet(nn.Module):

    def __init__(self, init_nfeatures=4, growth_rate=2, reduction=0.5, nlayers=[1, 2, 2, 2, 2, 2, 1],
                 dilation_rates=[3, 6, 12, 18, 24], dense_growth_rate=4):
        super(DenseNet_DenseASPP_FullVNet, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_features_1 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_features_2 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_features_3 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features

        self.denseaspp = DenseASPP_module(num_features, dense_growth_rate, dilation_rates)

        num_features = dense_growth_rate
        num_out_features = (num_features + num_features_3) // 2
        self.up3 = UpTrans(num_features, num_features_3, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup3 = DenseBlock(nlayers[4], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[4] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup3 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_2) // 2
        self.up2 = UpTrans(num_features, num_features_2, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup2 = DenseBlock(nlayers[5], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[5] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup2 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_1) // 2
        self.up1 = UpTrans(num_features, num_features_1, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup1 = DenseBlock(nlayers[6], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[6] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans1 = Trans(num_features, num_out_features, drop_rate=0.2)

        self.output = Output(num_out_features)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        trans4 = self.trans4(dense4)

        denseaspp = self.denseaspp(trans4)

        up3 = self.up3(denseaspp, dense3)
        denseup3 = self.denseup3(up3)
        transup3 = self.transup3(denseup3)

        up2 = self.up2(transup3, dense2)
        denseup2 = self.denseup2(up2)
        transup2 = self.transup2(denseup2)

        up1 = self.up1(transup2, dense1)
        denseup1 = self.denseup1(up1)
        trans1 = self.trans1(denseup1)

        out = self.output(trans1)

        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(down1.size())
            print(dense2.size())
            print(down2.size())
            print(dense3.size())
            print(down3.size())
            print(dense4.size())
            print(trans4.size())
            print(denseaspp.size())
            print(up3.size())
            print(denseup3.size())
            print(transup3.size())
            print(up2.size())
            print(denseup2.size())
            print(transup2.size())
            print(up1.size())
            print(denseup1.size())
            print(trans1.size())
            print(out.size())

        return out


class DenseNet_DenseASPP_SimpleVNet(nn.Module):

    def __init__(self, init_nfeatures=4, growth_rate=4, reduction=0.5, nlayers=[2, 3, 4, 4],
                 dilation_rates=[3, 6, 12, 18, 24], dense_growth_rate=8):
        super(DenseNet_DenseASPP_SimpleVNet, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_features_0 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_features_1 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_features_2 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features

        self.denseaspp = DenseASPP_module(num_features, dense_growth_rate, dilation_rates)

        num_features = dense_growth_rate
        num_out_features = (num_features + num_features_2) // 2
        self.up3 = UpTrans(num_features, num_features_2, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_1) // 2
        self.up2 = UpTrans(num_features, num_features_1, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_0) // 2
        self.up1 = UpTrans(num_features, num_features_0, num_out_features, drop_rate=0.2)

        self.output = Output(num_out_features)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        trans4 = self.trans4(dense4)

        denseaspp = self.denseaspp(trans4)

        up3 = self.up3(denseaspp, dense3)

        up2 = self.up2(up3, dense2)

        up1 = self.up1(up2, dense1)
        out = self.output(up1)

        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(down1.size())
            print(dense2.size())
            print(down2.size())
            print(dense3.size())
            print(down3.size())
            print(dense4.size())
            print(trans4.size())
            print(denseaspp.size())
            print(up3.size())
            print(up2.size())
            print(up1.size())
            print(out.size())

        return out


class DenseASPP_module(nn.Module):
    def __init__(self, num_input_features, dense_growth_rate, dilation_rates):
        super(DenseASPP_module, self).__init__()

        num_features = num_input_features
        self.ASPP_3 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[0], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_6 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[1], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_12 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[2], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_18 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[3], drop_out=0.2)
        num_features += dense_growth_rate

        self.ASPP_24 = DenseAsppBlock(num_features, dense_growth_rate, dilation_rate=dilation_rates[4], drop_out=0.2)
        num_features += dense_growth_rate

        self.trans = nn.Conv3d(num_features, dense_growth_rate, kernel_size=1, bias=False)

    def forward(self, x):

        aspp3 = self.ASPP_3(x)
        aspp6 = self.ASPP_6(aspp3)
        aspp12 = self.ASPP_12(aspp6)
        aspp18 = self.ASPP_18(aspp12)
        aspp24 = self.ASPP_24(aspp18)

        return self.trans(aspp24)


class DenseNet_DenseASPP(nn.Module):

    def __init__(self, init_nfeatures=8, growth_rate=4, reduction=0.5, nlayers=[2, 3, 4, 4],
                 dilation_rates=[3, 6, 12, 18, 24], dense_growth_rate=8):
        super(DenseNet_DenseASPP, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features

        self.denseaspp = DenseASPP_module(num_features, dense_growth_rate, dilation_rates)

        self.trans0 = Trans(dense_growth_rate, 2, drop_rate=0.2)

        self.upsample = nn.Upsample(scale_factor=8, mode='trilinear')

        self.output = Output2(2)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        trans4 = self.trans4(dense4)

        denseaspp = self.denseaspp(trans4)

        trans0 = self.trans0(denseaspp)

        up = self.upsample(trans0)

        out = self.output(up)
        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(dense2.size())
            print(dense3.size())
            print(dense4.size())
            print(denseaspp.size())
            print(trans0.size())
            print(up.size())
            print(out.size())
        return out


class ASPP_module(nn.Module):
    def __init__(self, num_input_features, num_out_features, dilation_rates):
        super(ASPP_module, self).__init__()

        self.input = nn.Sequential(nn.BatchNorm3d(num_input_features), nn.ReLU(inplace=True))

        self.conv11_0 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=1,
                                                padding=0, dilation=dilation_rates[0], bias=False),
                                      nn.BatchNorm3d(num_out_features))
        self.conv33_1 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=3,
                                                padding=dilation_rates[1], dilation=dilation_rates[1], bias=False),
                                      nn.BatchNorm3d(num_out_features))
        self.conv33_2 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=3,
                                                padding=dilation_rates[2], dilation=dilation_rates[2], bias=False),
                                      nn.BatchNorm3d(num_out_features))
        self.conv33_3 = nn.Sequential(nn.Conv3d(num_input_features, num_out_features, kernel_size=3,
                                                padding=dilation_rates[3], dilation=dilation_rates[3], bias=False),
                                      nn.BatchNorm3d(num_out_features))

        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_avg = nn.Conv3d(num_input_features, num_out_features, kernel_size=1, bias=False)

        self.num_input_features = num_input_features
        self.num_out_features = num_out_features

        self.cat_conv = nn.Conv3d(num_out_features * 5, num_out_features, kernel_size=1, bias=False)

    def forward(self, x):

        input = self.input(x)

        conv11_0 = self.conv11_0(input)
        conv33_1 = self.conv33_1(input)
        conv33_2 = self.conv33_2(input)
        conv33_3 = self.conv33_3(input)

        avg = self.global_avg_pool(input)
        conv_avg = self.conv_avg(avg)
        upsample = F.upsample(conv_avg, size=x.size()[2:], mode='trilinear', align_corners=True)

        concate = torch.cat((conv11_0, conv33_1, conv33_2, conv33_3, upsample), 1)

        return self.cat_conv(concate)


class DenseNet_ASPP(nn.Module):

    def __init__(self, init_nfeatures=32, growth_rate=16, reduction=0.5, nlayers=[2, 3, 4, 4],
                 dilation_rates=[1, 6, 12, 18], aspp_channels=32):
        super(DenseNet_ASPP, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features

        self.aspp = ASPP_module(num_features, aspp_channels, dilation_rates)

        self.trans5 = Trans(aspp_channels, 2, drop_rate=0.2)

        self.upsample = nn.Upsample(scale_factor=8, mode='trilinear')

        self.output = Output2(2)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        trans4 = self.trans4(dense4)

        aspp = self.aspp(trans4)

        trans5 = self.trans5(aspp)

        up = self.upsample(trans5)

        out = self.output(up)

        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(down1.size())
            print(dense2.size())
            print(down2.size())
            print(dense3.size())
            print(down3.size())
            print(dense4.size())
            print(trans4.size())
            print(aspp.size())
            print(trans5.size())
            print(up.size())
            print(out.size())

        return out


class DenseNet_ASPP_SimpleVNet(nn.Module):

    def __init__(self, init_nfeatures=16, growth_rate=8, reduction=0.5, nlayers=[2, 3, 4, 4],
                 dilation_rates=[1, 6, 12, 18], aspp_channels=32):
        super(DenseNet_ASPP_SimpleVNet, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_features_0 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_features_1 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_features_2 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features

        self.aspp = ASPP_module(num_features, aspp_channels, dilation_rates)

        num_features = aspp_channels
        num_out_features = (num_features + num_features_2) // 2
        self.up3 = UpTrans(num_features, num_features_2, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_1) // 2
        self.up2 = UpTrans(num_features, num_features_1, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_0) // 2
        self.up1 = UpTrans(num_features, num_features_0, num_out_features, drop_rate=0.2)

        self.output = Output(num_out_features)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        trans4 = self.trans4(dense4)

        aspp = self.aspp(trans4)

        up3 = self.up3(aspp, dense3)

        up2 = self.up2(up3, dense2)

        up1 = self.up1(up2, dense1)
        out = self.output(up1)

        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(down1.size())
            print(dense2.size())
            print(down2.size())
            print(dense3.size())
            print(down3.size())
            print(dense4.size())
            print(trans4.size())
            print(aspp.size())
            print(up3.size())
            print(up2.size())
            print(up1.size())
            print(out.size())

        return out


class DenseNet_ASPP_FullVNet(nn.Module):

    def __init__(self, init_nfeatures=4, growth_rate=2, reduction=0.5, nlayers=[1, 2, 2, 2, 2, 2, 1], dilation_rates=[1, 6, 12, 18], aspp_channels=8):
        super(DenseNet_ASPP_FullVNet, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_features_1 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_features_2 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_features_3 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features

        self.aspp = ASPP_module(num_features, aspp_channels, dilation_rates)

        num_features = aspp_channels
        num_out_features = (num_features + num_features_3) // 2
        self.up3 = UpTrans(num_features, num_features_3, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup3 = DenseBlock(nlayers[4], num_features, growth_rate, dilation_rate=4, drop_rate=0.2)
        num_features += nlayers[4] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup3 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_2) // 2
        self.up2 = UpTrans(num_features, num_features_2, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup2 = DenseBlock(nlayers[5], num_features, growth_rate, dilation_rate=2, drop_rate=0.2)
        num_features += nlayers[5] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup2 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_1) // 2
        self.up1 = UpTrans(num_features, num_features_1, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup1 = DenseBlock(nlayers[6], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[6] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans1 = Trans(num_features, num_out_features, drop_rate=0.2)

        self.output = Output(num_out_features)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        trans4 = self.trans4(dense4)

        aspp = self.aspp(trans4)

        up3 = self.up3(aspp, dense3)
        denseup3 = self.denseup3(up3)
        transup3 = self.transup3(denseup3)

        up2 = self.up2(transup3, dense2)
        denseup2 = self.denseup2(up2)
        transup2 = self.transup2(denseup2)

        up1 = self.up1(transup2, dense1)
        denseup1 = self.denseup1(up1)
        trans1 = self.trans1(denseup1)

        out = self.output(trans1)

        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(down1.size())
            print(dense2.size())
            print(down2.size())
            print(dense3.size())
            print(down3.size())
            print(dense4.size())
            print(trans4.size())
            print(aspp.size())
            print(up3.size())
            print(denseup3.size())
            print(transup3.size())
            print(up2.size())
            print(denseup2.size())
            print(transup2.size())
            print(up1.size())
            print(denseup1.size())
            print(trans1.size())
            print(out.size())

        return out


class DenseNet_FullVNet(nn.Module):

    def __init__(self, init_nfeatures=4, growth_rate=2, reduction=0.5, nlayers=[1, 2, 2, 2, 2, 2, 2, 2, 1]):
        super(DenseNet_FullVNet, self).__init__()

        self.input = Input(init_nfeatures)

        num_features = init_nfeatures
        self.dense1 = DenseBlock(nlayers[0], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[0] * growth_rate
        num_features_1 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down1 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense2 = DenseBlock(nlayers[1], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[1] * growth_rate
        num_features_2 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down2 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense3 = DenseBlock(nlayers[2], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[2] * growth_rate
        num_features_3 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down3 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense4 = DenseBlock(nlayers[3], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[3] * growth_rate
        num_features_4 = num_features
        num_out_features = int(math.floor(num_features * reduction))
        self.down4 = DownTrans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.dense5 = DenseBlock(nlayers[4], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[4] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans5 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_4) // 2
        self.up4 = UpTrans(num_features, num_features_4, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup4 = DenseBlock(nlayers[5], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[5] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup4 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_3) // 2
        self.up3 = UpTrans(num_features, num_features_3, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup3 = DenseBlock(nlayers[6], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[6] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup3 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_2) // 2
        self.up2 = UpTrans(num_features, num_features_2, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup2 = DenseBlock(nlayers[7], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[7] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.transup2 = Trans(num_features, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        num_out_features = (num_features + num_features_1) // 2
        self.up1 = UpTrans(num_features, num_features_1, num_out_features, drop_rate=0.2)

        num_features = num_out_features
        self.denseup1 = DenseBlock(nlayers[8], num_features, growth_rate, drop_rate=0.2)
        num_features += nlayers[8] * growth_rate
        num_out_features = int(math.floor(num_features * reduction))
        self.trans1 = Trans(num_features, num_out_features, drop_rate=0.2)

        self.output = Output(num_out_features)

    def forward(self, x):

        input = self.input(x)

        dense1 = self.dense1(input)
        down1 = self.down1(dense1)

        dense2 = self.dense2(down1)
        down2 = self.down2(dense2)

        dense3 = self.dense3(down2)
        down3 = self.down3(dense3)

        dense4 = self.dense4(down3)
        down4 = self.down4(dense4)

        dense5 = self.dense5(down4)
        trans5 = self.trans5(dense5)

        up4 = self.up4(trans5, dense4)
        denseup4 = self.denseup4(up4)
        transup4 = self.transup4(denseup4)

        up3 = self.up3(transup4, dense3)
        denseup3 = self.denseup3(up3)
        transup3 = self.transup3(denseup3)

        up2 = self.up2(transup3, dense2)
        denseup2 = self.denseup2(up2)
        transup2 = self.transup2(denseup2)

        up1 = self.up1(transup2, dense1)
        denseup1 = self.denseup1(up1)
        trans1 = self.trans1(denseup1)

        out = self.output(trans1)

        if False:
            print(x.size())
            print(input.size())
            print(dense1.size())
            print(down1.size())
            print(dense2.size())
            print(down2.size())
            print(dense3.size())
            print(down3.size())
            print(dense4.size())
            print(trans4.size())
            print(up3.size())
            print(denseup3.size())
            print(transup3.size())
            print(up2.size())
            print(denseup2.size())
            print(transup2.size())
            print(up1.size())
            print(denseup1.size())
            print(trans1.size())
            print(out.size())

        return out
