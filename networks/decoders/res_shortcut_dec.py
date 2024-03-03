from networks.decoders.resnet_dec import ResNet_D_Dec
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBNRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_relu=True):
        super(Conv2dBNRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True)
        ]

        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GFA(nn.Module):
    def __init__(self, in_planes):
        super(GFA, self).__init__()

        self.conv1 = nn.Conv2d(512, in_planes, kernel_size=1, bias=False)
        self.convbnrelu = Conv2dBNRelu(in_planes * 2, in_planes)
        self.convbnsigmoid = Conv2dBNRelu(in_planes, in_planes, kernel_size=1, padding=0, with_relu=False)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=1, bias=False)

    def forward(self, se_fea, detail_fea):
        se_fea = self.conv1(se_fea)
        guide_fea = torch.cat([se_fea, detail_fea], dim=1)
        guide_fea = self.convbnrelu(guide_fea)
        guide_fea = self.convbnsigmoid(guide_fea)
        guide_fea = guide_fea * detail_fea + detail_fea
        out = self.conv2(guide_fea)

        return out


class ResShortCut_D_Dec(ResNet_D_Dec):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        img = mid_fea['image']
        se_fea = x

        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        pred_os8 = self.refine_OS8(x)

        x = self.layer3(x) + fea3
        pred_os4 = self.refine_OS4(x)

        x = self.layer4(x) + fea2
        pred_os2 = self.refine_OS2(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        pred_matte = self.refine_OS1(torch.cat((x, img), dim=1))

        pred_os8 = F.interpolate(pred_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        pred_os4 = F.interpolate(pred_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        pred_os2 = F.interpolate(pred_os2, scale_factor=2.0, mode='bilinear', align_corners=False)

        pred_os8 = (torch.tanh(pred_os8) + 1.0) / 2.0
        pred_os4 = (torch.tanh(pred_os4) + 1.0) / 2.0
        pred_os2 = (torch.tanh(pred_os2) + 1.0) / 2.0
        pred_alpha = (torch.tanh(pred_matte) + 1.0) / 2.0

        return pred_alpha, pred_os2, pred_os4, pred_os8
