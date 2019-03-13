import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLayer(nn.Module):
    def __init__(self, num_features, drop_rate=0, dilation_rate=1):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv3d(num_features, num_features, kernel_size=3, dilation=dilation_rate, padding=dilation_rate, bias=False)
        self.bn = nn.BatchNorm3d(num_features)
        self.relu = nn.PReLU(num_features)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out

class ResBlock(nn.Module):
    def __init__(self, num_features, num_layers, drop_rate=0, dilation_rate=1):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(int(num_layers)):
            layers.append(SingleLayer(num_features, drop_rate, dilation_rate=dilation_rate))
        self.layers = nn.Sequential(*layers)
        self.relu = nn.PReLU(num_features)

    def forward(self, x):
        out = self.layers(x)
        out = self.relu(torch.add(out, x))
        return out

class DownTrans(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate=0, dilation = 1):
        super(DownTrans, self).__init__()
        self.down_conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=2, stride=2, bias=False, dilation = dilation)
        self.bn = nn.BatchNorm3d(num_output_features)

        self.relu = nn.PReLU(num_output_features)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.relu(self.bn(self.down_conv(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out

class UpTrans(nn.Module):
    def __init__(self, num_input_features, num_out_features, drop_rate=0, dilation = 1):
        super(UpTrans, self).__init__()

        self.up_conv1 = nn.ConvTranspose3d(num_input_features, num_out_features // 2, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(num_out_features // 2)

        self.relu = nn.PReLU(num_out_features // 2)

        self.drop_rate = drop_rate

    def forward(self, x, skip):
        out = self.relu(self.bn1(self.up_conv1(x)))
        out = torch.cat((out, skip), 1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out

class Input(nn.Module):
    def __init__(self, num_out_features, dilation = 1):
        super(Input, self).__init__()
        self.conv = nn.Conv3d(1, num_out_features, kernel_size=3, padding=1, bias=False, dilation = dilation)
        self.bn = nn.BatchNorm3d(num_out_features)
        self.relu = nn.PReLU(num_out_features)

    def forward(self, x):
        out = self.bn(self.conv(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        out = self.relu(torch.add(out, x16))
        return out

class Input8(nn.Module):
    def __init__(self, num_out_features):
        super(Input8, self).__init__()
        self.conv = nn.Conv3d(1, num_out_features, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm3d(num_out_features)
        self.relu = nn.PReLU(num_out_features)

    def forward(self, x):
        out = self.bn(self.conv(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x), 1)
        out = self.relu(torch.add(out, x16))
        return out

class Output(nn.Module):
    def __init__(self, num_input_features, out_channel):
        super(Output, self).__init__()
        self.conv1 = nn.Conv3d(num_input_features, out_channels = out_channel, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(in_channels = out_channel, out_channels = out_channel, kernel_size=1)
        self.relu1 = nn.PReLU(out_channel)

        #self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 4 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        #out = out.permute(0, 2, 3, 4, 1).contiguous()
        #out = out.view(out.numel() // out_channel, out_channel)
        #out = self.softmax(out, 1)
        return out

class Output_segments(nn.Module):
    def __init__(self, num_input_features):
        out_channel = 6
        super(Output_segments, self).__init__()
        self.conv1 = nn.Conv3d(num_input_features, out_channel, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=1)
        self.relu1 = nn.PReLU(out_channel)

        self.softmax = F.softmax

    def forward(self, x):
        out_channel = 6
        # convolve 32 down to 4 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # print(out.size())
        # flatten
        out = out.view(out.numel() // out_channel, out_channel)
        out = self.softmax(out, 1)
        # treat channel 0 as the predicted output
        return out

class ResVNet(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features = num_init_features*2, out_channel = 5)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class ResVNet_Triplet(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_Triplet, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_input_features = num_init_features*2, out_channel = 4)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class ResVNet_Triplet_Try(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_Triplet_Try, self).__init__()

        self.input = Input(num_init_features, dilation = [2, 1, 1])

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0], dilation = [2, 1, 1])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4, )
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1], dilation = [2, 1, 1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2, dilation = [2, 1, 1])
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2, dilation = [2, 1, 1])
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2, dilation = [2, 1, 1])

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2, dilation = [2, 1, 1])
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2, dilation = [2, 1, 1])
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4, dilation = [2, 1, 1])
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2, dilation = [2, 1, 1])
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_input_features = num_init_features*2, out_channel = 4)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class ResVNet_Heart(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_Heart, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_input_features = num_init_features*2, out_channel = 2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class DeconvLayer(nn.Module):
    def __init__(self, num_input_features, num_out_features):
        super(DeconvLayer, self).__init__()
        self.deconv =  nn.ConvTranspose3d(num_input_features, num_out_features, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(num_out_features)
        self.relu = nn.PReLU(num_out_features)

    def forward(self, x):
        out = self.relu(self.bn(self.deconv(x)))
        return out

class SupOut(nn.Module):
    def __init__(self, num_input_features, nDeconvs):
        super(SupOut, self).__init__()
        self.conv1 = nn.Conv3d(num_input_features, 2, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm3d(2)

        layers = []
        for i in range(int(nDeconvs)):
            layers.append(DeconvLayer(2, 2))
        self.layers = nn.Sequential(*layers)

        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = nn.PReLU(2)

        self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out, 1)
        # treat channel 0 as the predicted output
        return out

class ResVNet_DeepSup(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_DeepSup, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.deepsup7 = SupOut(num_init_features*2, 1)

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.deepsup6 =  SupOut(num_init_features*4, 2)

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.deepsup5 = SupOut(num_init_features*8, 3)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.deepsup4 = SupOut(num_init_features*16, 4)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.deepsup3 = SupOut(num_init_features*16, 3)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.deepsup2 = SupOut(num_init_features*8, 2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.deepsup1 = SupOut(num_init_features*4, 1)

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        supout7 = self.deepsup7(resdown1)
        supout6 = self.deepsup6(resdown2)
        supout5 = self.deepsup5(resdown3)
        supout4 = self.deepsup4(resdown4)
        supout3 = self.deepsup3(resup4)
        supout2 = self.deepsup2(resup3)
        supout1 = self.deepsup1(resup2)

        return supout7, supout6, supout5, supout4, supout3, supout2, supout1, out

class ResVNet8(nn.Module):
    def __init__(self, num_init_features=8, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet8, self).__init__()

        self.input = Input8(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class ResVNet8_DeepSup(nn.Module):
    def __init__(self, num_init_features=4, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet8_DeepSup, self).__init__()

        self.input = Input4(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.deepsup7 = SupOut(num_init_features*2, 1)

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.deepsup6 =  SupOut(num_init_features*4, 2)

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)

        self.deepsup5 = SupOut(num_init_features*8, 3)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)

        self.deepsup4 = SupOut(num_init_features*16, 4)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.deepsup3 = SupOut(num_init_features*16, 3)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.deepsup2 = SupOut(num_init_features*8, 2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.deepsup1 = SupOut(num_init_features*4, 1)

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)

        up4 = self.up4(resdown4, resdown3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        supout7 = self.deepsup7(resdown1)
        supout6 = self.deepsup6(resdown2)
        supout5 = self.deepsup5(resdown3)
        supout4 = self.deepsup4(resdown4)
        supout3 = self.deepsup3(resup4)
        supout2 = self.deepsup2(resup3)
        supout1 = self.deepsup1(resup2)

        return supout7, supout6, supout5, supout4, supout3, supout2, supout1, out

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

        self.global_avg_pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
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
        #upsample = F.upsample(conv_avg, size=x.size()[2:], mode='trilinear')
        upsample = F.interpolate(conv_avg, size = x.size()[2:], mode = 'trilinear', align_corners = True)

        concate = torch.cat((conv11_0, conv33_1, conv33_2, conv33_3, upsample), 1)

        return self.cat_conv(concate)

class ResVNet_ASPP_Triplet(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1],
                 dilation_rates=[1, 6, 12, 18]):
        super(ResVNet_ASPP_Triplet, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], dilation_rate=2, drop_rate=0.2)

        self.aspp = ASPP_module(num_init_features*8, num_init_features*8, dilation_rates)

        self.up3 = UpTrans(num_init_features*8, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], dilation_rate=2, drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_input_features = num_init_features*2, out_channel = 4)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        aspp = self.aspp(resdown3)

        up3 = self.up3(aspp, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        #out = out.contiguous().torch.(1, 16, 48, 48, 5)
        #print("in the resvnet_aspp out.size() = ", out.size())
        return out

class ResVNet_ASPP_Heart(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1],
                 dilation_rates=[1, 6, 12, 18]):
        super(ResVNet_ASPP_Bin, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], dilation_rate=2, drop_rate=0.2)

        self.aspp = ASPP_module(num_init_features*8, num_init_features*8, dilation_rates)

        self.up3 = UpTrans(num_init_features*8, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], dilation_rate=2, drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output_bin = Output_Bin(num_input_features = num_init_features*2, out_channel = 2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        aspp = self.aspp(resdown3)

        up3 = self.up3(aspp, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output_bin(resup1)
        return out

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

        self.trans = nn.Conv3d(num_features, num_input_features, kernel_size=1, bias=False)

    def forward(self, x):

        aspp3 = self.ASPP_3(x)
        aspp6 = self.ASPP_6(aspp3)
        aspp12 = self.ASPP_12(aspp6)
        aspp18 = self.ASPP_18(aspp12)
        aspp24 = self.ASPP_24(aspp18)

        return self.trans(aspp24)

class ResVNet_DenseASPP(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1],
                 dilation_rates=[3, 6, 12, 18, 24], dense_growth_rate=64):
        super(ResVNet_DenseASPP, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], dilation_rate=2, drop_rate=0.2)

        self.denseaspp = DenseASPP_module(num_init_features*8, dense_growth_rate, dilation_rates)

        self.up3 = UpTrans(num_init_features*8, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], dilation_rate=2, drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)

        denseaspp = self.denseaspp(resdown3)

        up3 = self.up3(denseaspp, resdown2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, resdown1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class GCN(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(GCN, self).__init__()

        k = 7
        pad = 3

        self.conv_11 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(k, 1, 1), padding=(pad, 0, 0), bias=False)
        self.conv_12 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(1, k, 1), padding=(0, pad, 0), bias=False)
        self.conv_13 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(1, 1, k), padding=(0, 0, pad), bias=False)

        self.conv_21 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(1, k, 1), padding=(0, pad, 0), bias=False)
        self.conv_22 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(1, 1, k), padding=(0, 0, pad), bias=False)
        self.conv_23 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(k, 1, 1), padding=(pad, 0, 0), bias=False)

        self.conv_31 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(1, 1, k), padding=(0, 0, pad), bias=False)
        self.conv_32 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(k, 1, 1), padding=(pad, 0, 0), bias=False)
        self.conv_33 = nn.Conv3d(num_input_features, num_output_features, kernel_size=(1, k, 1), padding=(0, pad, 0), bias=False)

    def forward(self, x):
        x1 = self.conv_11(x)
        x1 = self.conv_12(x1)
        x1 = self.conv_13(x1)

        x2 = self.conv_21(x)
        x2 = self.conv_22(x2)
        x2 = self.conv_23(x2)

        x3 = self.conv_31(x)
        x3 = self.conv_32(x3)
        x3 = self.conv_33(x3)

        out = x1 + x2 + x3

        return out

class ResVNet_GCN(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_GCN, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])
        self.gcn1 = GCN(num_init_features*2, num_init_features*2)

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])
        self.gcn2 = GCN(num_init_features*4, num_init_features*4)

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)
        self.gcn3 = GCN(num_init_features*8, num_init_features*8)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)
        self.gcn4 = GCN(num_init_features*16, num_init_features*16)

        self.up4 = UpTrans(num_init_features*16, num_init_features*16, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTrans(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTrans(num_init_features*8, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTrans(num_init_features*4, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)
        gcn1 = self.gcn1(resdown1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)
        gcn2 = self.gcn2(resdown2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)
        gcn3 = self.gcn3(resdown3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)
        gcn4 = self.gcn4(resdown4)

        up4 = self.up4(gcn4, gcn3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, gcn2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, gcn1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class UpTransAdd(nn.Module):
    def __init__(self, num_input_features, num_out_features, drop_rate=0):
        super(UpTransAdd, self).__init__()

        self.up_conv1 = nn.ConvTranspose3d(num_input_features, num_out_features, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(num_out_features)

        self.relu = nn.PReLU(num_out_features)

        self.drop_rate = drop_rate

    def forward(self, x, skip):
        out = self.relu(self.bn1(self.up_conv1(x)))
        out = torch.add(out, skip)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out

class ResVNet_GCN_Add(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_GCN_Add, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])
        self.gcn1 = GCN(num_init_features*2, num_init_features*2)

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])
        self.gcn2 = GCN(num_init_features*4, num_init_features*4)

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)
        self.gcn3 = GCN(num_init_features*8, num_init_features*8)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)
        self.gcn4 = GCN(num_init_features*16, num_init_features*16)

        self.up4 = UpTransAdd(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.resup4 = ResBlock(num_init_features*8, nlayers[4], drop_rate=0.2)

        self.up3 = UpTransAdd(num_init_features*8, num_init_features*4, drop_rate=0.2)
        self.resup3 = ResBlock(num_init_features*4, nlayers[5], drop_rate=0.2)

        self.up2 = UpTransAdd(num_init_features*4, num_init_features*2)
        self.resup2 = ResBlock(num_init_features*2, nlayers[6])

        self.up1 = UpTransAdd(num_init_features*2, num_init_features*1)
        self.resup1 = ResBlock(num_init_features*1, nlayers[7])

        self.output = Output(num_init_features*1)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)
        gcn1 = self.gcn1(resdown1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)
        gcn2 = self.gcn2(resdown2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)
        gcn3 = self.gcn3(resdown3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)
        gcn4 = self.gcn4(resdown4)

        up4 = self.up4(gcn4, gcn3)
        resup4 = self.resup4(up4)

        up3 = self.up3(resup4, gcn2)
        resup3 = self.resup3(up3)

        up2 = self.up2(resup3, gcn1)
        resup2 = self.resup2(up2)

        up1 = self.up1(resup2, input)
        resup1 = self.resup1(up1)

        out = self.output(resup1)

        return out

class CAB(nn.Module):
    def __init__(self, num_input_features, num_out_features):
        super(CAB, self).__init__()

        self.global_avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(num_input_features, num_out_features // 2, kernel_size=1, bias=False)
        self.relu  = nn.PReLU(num_out_features // 2)
        self.conv2 = nn.Conv3d(num_out_features // 2, num_out_features // 2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skipx):

        incat = torch.cat((x, skipx), 1)
        w = self.sigmoid(self.conv2(self.relu(self.conv1(self.global_avg_pool(incat)))))

        skipx = skipx * w

        out = torch.cat((x, skipx), 1)

        return out

class UpTransNoCat(nn.Module):
    def __init__(self, num_input_features, num_out_features, drop_rate=0):
        super(UpTransNoCat, self).__init__()

        self.up_conv1 = nn.ConvTranspose3d(num_input_features, num_out_features, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(num_out_features)

        self.relu = nn.PReLU(num_out_features)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.relu(self.bn1(self.up_conv1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate)
        return out

class ResVNet_GCN_CAB(nn.Module):
    def __init__(self, num_init_features=16, nlayers=[2, 3, 3, 3, 3, 3, 2, 1]):
        super(ResVNet_GCN_CAB, self).__init__()

        self.input = Input(num_init_features)

        self.down1 = DownTrans(num_init_features, num_init_features*2)
        self.resdown1 = ResBlock(num_init_features*2, nlayers[0])
        self.gcn1 = GCN(num_init_features*2, num_init_features*2)

        self.down2 = DownTrans(num_init_features*2, num_init_features*4)
        self.resdown2 = ResBlock(num_init_features*4, nlayers[1])
        self.gcn2 = GCN(num_init_features*4, num_init_features*4)

        self.down3 = DownTrans(num_init_features*4, num_init_features*8, drop_rate=0.2)
        self.resdown3 = ResBlock(num_init_features*8, nlayers[2], drop_rate=0.2)
        self.gcn3 = GCN(num_init_features*8, num_init_features*8)

        self.down4 = DownTrans(num_init_features*8, num_init_features*16, drop_rate=0.2)
        self.resdown4 = ResBlock(num_init_features*16, nlayers[3], drop_rate=0.2)
        self.gcn4 = GCN(num_init_features*16, num_init_features*16)

        self.up4 = UpTransNoCat(num_init_features*16, num_init_features*8, drop_rate=0.2)
        self.cab4 = CAB(num_init_features*16, num_init_features*16)
        self.resup4 = ResBlock(num_init_features*16, nlayers[4], drop_rate=0.2)

        self.up3 = UpTransNoCat(num_init_features*16, num_init_features*4, drop_rate=0.2)
        self.cab3 = CAB(num_init_features*8, num_init_features*8)
        self.resup3 = ResBlock(num_init_features*8, nlayers[5], drop_rate=0.2)

        self.up2 = UpTransNoCat(num_init_features*8, num_init_features*2)
        self.cab2 = CAB(num_init_features*4, num_init_features*4)
        self.resup2 = ResBlock(num_init_features*4, nlayers[6])

        self.up1 = UpTransNoCat(num_init_features*4, num_init_features*1)
        self.cab1 = CAB(num_init_features*2, num_init_features*2)
        self.resup1 = ResBlock(num_init_features*2, nlayers[7])

        self.output = Output(num_init_features*2)

    def forward(self, x):

        input = self.input(x)

        down1 = self.down1(input)
        resdown1 = self.resdown1(down1)
        gcn1 = self.gcn1(resdown1)

        down2 = self.down2(resdown1)
        resdown2 = self.resdown2(down2)
        gcn2 = self.gcn2(resdown2)

        down3 = self.down3(resdown2)
        resdown3 = self.resdown3(down3)
        gcn3 = self.gcn3(resdown3)

        down4 = self.down4(resdown3)
        resdown4 = self.resdown4(down4)
        gcn4 = self.gcn4(resdown4)

        up4 = self.up4(gcn4)
        cab4 = self.cab4(up4, gcn3)
        resup4 = self.resup4(cab4)

        up3 = self.up3(resup4)
        cab3 = self.cab3(up3, gcn2)
        resup3 = self.resup3(cab3)

        up2 = self.up2(resup3)
        cab2 = self.cab2(up2, gcn1)
        resup2 = self.resup2(cab2)

        up1 = self.up1(resup2)
        cab1 = self.cab1(up1, input)
        resup1 = self.resup1(cab1)

        out = self.output(resup1)

        return out
