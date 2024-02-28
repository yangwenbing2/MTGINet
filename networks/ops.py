import torch
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


class ASPP(nn.Module):
    '''
    based on https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
    '''

    def __init__(self, in_channel, out_channel, conv=nn.Conv2d, norm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        mid_channel = 256
        dilations = [1, 2, 4, 8]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(in_channel, mid_channel, kernel_size=1, stride=1, dilation=dilations[0], bias=False)

        self.aspp2 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                          dilation=dilations[1], padding=dilations[1],
                          bias=False)
        self.aspp3 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                          dilation=dilations[2], padding=dilations[2],
                          bias=False)
        self.aspp4 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                          dilation=dilations[3], padding=dilations[3],
                          bias=False)

        self.aspp5 = conv(in_channel, mid_channel, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(mid_channel)
        self.aspp2_bn = norm(mid_channel)
        self.aspp3_bn = norm(mid_channel)
        self.aspp4_bn = norm(mid_channel)
        self.aspp5_bn = norm(mid_channel)
        self.conv2 = conv(mid_channel * 5, out_channel, kernel_size=1, stride=1,
                          bias=False)
        self.bn2 = norm(out_channel)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='nearest')(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _LargeKernelConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_LargeKernelConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class AGCN(nn.Module):
    '''
    based on https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
    '''

    def __init__(self, in_channel, out_channel):
        super(AGCN, self).__init__()
        mid_channel = 256
        self.large_conv1 = _LargeKernelConvModule(in_channel, mid_channel, kernel_size=[7, 7])
        self.large_conv2 = _LargeKernelConvModule(in_channel, mid_channel, kernel_size=[11, 11])
        self.large_conv3 = _LargeKernelConvModule(in_channel, mid_channel, kernel_size=[15, 15])
        self.conv_fusion = nn.Conv2d(mid_channel * 3, out_channel, 1, stride=1, bias=False)

        self.bn_conv1 = nn.BatchNorm2d(mid_channel)
        self.bn_conv2 = nn.BatchNorm2d(mid_channel)
        self.bn_conv3 = nn.BatchNorm2d(mid_channel)
        self.bn_fusion = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.large_conv1(x)
        f1 = self.bn_conv1(f1)
        f1 = self.relu(f1)

        f2 = self.large_conv2(x)
        f2 = self.bn_conv2(f2)
        f2 = self.relu(f2)

        f3 = self.large_conv3(x)
        f3 = self.bn_conv3(f3)
        f3 = self.relu(f3)

        fusion = self.conv_fusion(torch.cat((f1, f2, f3), 1))
        fusion = self.bn_fusion(fusion)
        fusion = self.relu(fusion)

        out = fusion + x

        return out