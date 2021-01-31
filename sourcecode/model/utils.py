import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm = nn.BatchNorm2d

class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=atrous,
        dilation=atrous,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 atrous=1,
                 downsample=None,
                 norm_layer=None,
                 relu_inplace=True,
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.skip_add.add(out, residual)
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 atrous=1,
                 downsample=None,
                 reduction=16,
                 norm_layer=None,
                 relu_inplace=True,
                 **kwargs):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=atrous,
            dilation=atrous,
            bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.skip_add.add(out, residual)
        out = self.relu(out)
        return out


class SEBottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 atrous=1,
                 downsample=None,
                 reduction=16,
                 norm_layer=None,
                 strip_pool=False,
                 relu_inplace=True,
                 **kwargs):
        super(SEBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1 * atrous,
            dilation=atrous,
            bias=False)
        self.se = SELayer(planes * self.expansion, reduction)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride
        self.strip_pool = strip_pool
        self.spm = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se.forward(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 atrous=1,
                 downsample=None,
                 reduction=16,
                 norm_layer=None,
                 strip_pool=False,
                 relu_inplace=True,
                 **kwargs):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.spm is not None:
            out = out * self.spm(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptiveAvgPool2d(nn.Module):

    def __init__(self, tag=True, sz=None):
        super().__init__()
        self.sz = sz
        # TODO(zxyan): this is a magic number to avoid too large kernel.
        self.sz_list = range(0, 1000)
        self.tag = tag

    def forward(self, x):
        if self.training:
            if self.tag:
                return nn.AdaptiveAvgPool2d((None, 1))(x)
            else:
                return nn.AdaptiveAvgPool2d((1, None))(x)
        else:
            if torch.__version__ < '0.4.0':
                return F.avg_pool2d(
                    input=x,
                    ceil_mode=False,
                    kernel_size=(x.size(2), x.size(3)))
            else:
                if self.tag:
                    avg_pool2d = nn.AvgPool2d(
                        kernel_size=(1, self.sz_list.index(x.size(3))), )
                else:
                    avg_pool2d = nn.AvgPool2d(
                        kernel_size=(self.sz_list.index(x.size(2)), 1), )
                return avg_pool2d(x)


class ConvModule(nn.Module):
    """
    A conv block that contains conv/norm/activation layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 norm_layer=None,
                 activation='relu',
                 inplace=True):
        super(ConvModule, self).__init__()
        assert norm_layer is None or norm_layer == 'bn_2d' or norm_layer == 'sync_bn'
        self.activation = activation
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.add_module('0', conv)
        norm = BatchNorm(out_channels)
        self.add_module('1', norm)
        self.with_activation = activation is not None
        if self.with_activation:
            if self.activation == 'relu':
                activate = nn.ReLU(inplace=inplace)
                self.add_module('2', activate)
            else:
                raise ValueError

    def forward(self, x):
        x = self._modules['0'](x)
        x = self._modules['1'](x)
        if self.with_activation:
            x = self._modules['2'](x)

        return x


def build_resnet_block(block):
    return {
        'BasicBlock': BasicBlock,
        'BottleNeck': BottleNeck,
        'SEBasicBlock': SEBasicBlock,
        'SEBottleNeck': SEBottleNeck
    }[block]
