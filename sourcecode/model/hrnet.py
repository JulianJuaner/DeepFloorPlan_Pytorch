import torch
import torch.nn as nn
import torch.nn.functional as F
from sourcecode.model.utils import build_resnet_block, ConvModule
from sourcecode.utils.configs import merge_dict

BatchNorm = nn.BatchNorm2d

HRNetW18SmallV2_cfg = {
    'type': 'HRNetW18SmallV2',
    'in_channels': 3,
    'STAGE1': {
        'NUM_MODULES': 1,
        'NUM_RANCHES': 1,
        'BLOCK': 'BottleNeck',
        'NUM_BLOCKS': [2],
        'NUM_CHANNELS': [64],
        'FUSE_METHOD': 'SUM'
    },
    'STAGE2': {
        'NUM_MODULES': 1,
        'NUM_BRANCHES': 2,
        'BLOCK': 'BasicBlock',
        'NUM_BLOCKS': [2, 2],
        'NUM_CHANNELS': [18, 36],
        'FUSE_METHOD': 'SUM'
    },
    'STAGE3': {
        'NUM_MODULES': 3,
        'NUM_BRANCHES': 3,
        'BLOCK': 'BasicBlock',
        'NUM_BLOCKS': [2, 2, 2],
        'NUM_CHANNELS': [18, 36, 72],
        'FUSE_METHOD': 'SUM'
    },
    'STAGE4': {
        'NUM_MODULES': 2,
        'NUM_BRANCHES': 4,
        'BLOCK': 'BasicBlock',
        'NUM_BLOCKS': [2, 2, 2, 2],
        'NUM_CHANNELS': [18, 36, 72, 144],
        'FUSE_METHOD': 'SUM'
    }
}

class HighResolutionModule(nn.Module):

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 fuse_method,
                 norm_layer=None,
                 multi_scale_output=True,
                 strip_pool=False,
                 relu_inplace=True):
        super(HighResolutionModule, self).__init__()
        self.strip_pool = strip_pool
        self.trt_mode = False
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels,
                             num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            norm_layer,
            strip_pool=self.strip_pool,
            relu_inplace=relu_inplace)
        self.fuse_layers = self._make_fuse_layers(
            norm_layer, relu_inplace=relu_inplace)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.up = None
        self.skip_add = None

    def quantize(self):
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         norm_layer,
                         strip_pool=False,
                         stride=1,
                         relu_inplace=True):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = ConvModule(
                self.num_inchannels[branch_index],
                num_channels[branch_index] * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm_layer=norm_layer,
                activation=None)

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
                strip_pool=strip_pool,
                relu_inplace=relu_inplace))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    norm_layer=norm_layer,
                    strip_pool=strip_pool,
                    relu_inplace=relu_inplace))

        return nn.Sequential(*layers)

    def _make_branches(self,
                       num_branches,
                       block,
                       num_blocks,
                       num_channels,
                       norm_layer,
                       strip_pool=False,
                       relu_inplace=True):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    norm_layer,
                    strip_pool=strip_pool,
                    relu_inplace=relu_inplace))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, norm_layer, relu_inplace=True):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        ConvModule(
                            num_inchannels[j],
                            num_inchannels[i],
                            1,
                            1,
                            0,
                            bias=False,
                            norm_layer=norm_layer,
                            activation=None))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                ConvModule(
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    3,
                                    2,
                                    1,
                                    bias=False,
                                    norm_layer=norm_layer,
                                    activation=None))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                ConvModule(
                                    num_inchannels[j],
                                    num_outchannels_conv3x3,
                                    3,
                                    2,
                                    1,
                                    bias=False,
                                    norm_layer=norm_layer,
                                    activation='relu',
                                    inplace=relu_inplace))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    if self.skip_add:
                        y = self.skip_add.add(y, x[j])
                    else:
                        y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    
                    add_feature = F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')

                    if self.skip_add:
                        y = self.skip_add.add(y, add_feature)
                    else:
                        y = y + add_feature
                else:
                    if self.skip_add:
                        y = self.skip_add.add(y, self.fuse_layers[i][j](x[j]))
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):

    def __init__(self,
                 in_channels,
                 STAGE1,
                 STAGE2,
                 STAGE3,
                 STAGE4,
                 with_stp=False,
                 norm_layer=None,
                 relu_inplace=True,
                 **kwargs):
        super().__init__()

        # stem net
        self.input_channel = in_channels
        self.strip_pool = with_stp
        self.conv1 = nn.Conv2d(
            self.input_channel,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn1 = BatchNorm(64)
        self.conv2 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = STAGE1
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = build_resnet_block(self.stage1_cfg['BLOCK'])
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(
            block,
            64,
            num_channels,
            num_blocks,
            strip_pool=self.strip_pool,
            norm_layer=norm_layer,
            relu_inplace=relu_inplace)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = STAGE2
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = build_resnet_block(self.stage2_cfg['BLOCK'])
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel],
            num_channels,
            norm_layer,
            relu_inplace=relu_inplace)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg,
            num_channels,
            norm_layer,
            strip_pool=self.strip_pool,
            relu_inplace=relu_inplace)

        self.stage3_cfg = STAGE3
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = build_resnet_block(self.stage3_cfg['BLOCK'])
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
            norm_layer,
            relu_inplace=relu_inplace)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            norm_layer,
            strip_pool=self.strip_pool,
            relu_inplace=relu_inplace)

        self.stage4_cfg = STAGE4
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = build_resnet_block(self.stage4_cfg['BLOCK'])
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels,
            num_channels,
            norm_layer,
            relu_inplace=relu_inplace)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            norm_layer,
            multi_scale_output=True,
            strip_pool=self.strip_pool,
            relu_inplace=relu_inplace)

        # last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.per_stage_out_channels = pre_stage_channels

    def quantize(self):
        for m in self.modules():
            if isinstance(m, HighResolutionModule):
                m.quantize()

    def trt(self):
        for m in self.modules():
            if isinstance(m, HighResolutionModule):
                m.trt()

    def _make_transition_layer(self,
                               num_channels_pre_layer,
                               num_channels_cur_layer,
                               norm_layer=None,
                               relu_inplace=True):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        ConvModule(
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            3,
                            1,
                            1,
                            bias=False,
                            norm_layer=norm_layer,
                            activation='relu',
                            inplace=relu_inplace))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        ConvModule(
                            inchannels,
                            outchannels,
                            3,
                            2,
                            1,
                            bias=False,
                            norm_layer=norm_layer,
                            activation='relu',
                            inplace=relu_inplace))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self,
                    block,
                    inplanes,
                    planes,
                    blocks,
                    stride=1,
                    strip_pool=False,
                    norm_layer=None,
                    relu_inplace=True):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm_layer=norm_layer,
                activation=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                norm_layer=norm_layer,
                downsample=downsample,
                strip_pool=strip_pool,
                relu_inplace=relu_inplace))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    norm_layer=norm_layer,
                    relu_inplace=relu_inplace))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    layer_config,
                    num_inchannels,
                    norm_layer=None,
                    multi_scale_output=True,
                    strip_pool=False,
                    relu_inplace=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']

        block = build_resnet_block(layer_config['BLOCK'])
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    norm_layer,
                    reset_multi_scale_output,
                    strip_pool=strip_pool,
                    relu_inplace=relu_inplace))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def get_out_channels(self):
        return self.per_stage_out_channels

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        return x

class HRNetW18SmallV2(HighResolutionNet):

    def __init__(self, **kwargs):
        kwargs = merge_dict(HRNetW18SmallV2_cfg, kwargs)
        print(kwargs)
        super().__init__(**kwargs)

if "__main__" in __name__:
    HRNet = HRNetW18SmallV2().cuda()
    res = HRNet(torch.FloatTensor(torch.zeros([2, 3, 256, 256])).cuda())
    print(HRNet)
    print(res[0].shape)
