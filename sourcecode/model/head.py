import torch
import torch.nn as nn
import torch.nn.functional as F
from sourcecode.model.utils import SELayer


BatchNorm = nn.BatchNorm2d

class FloorHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.final_conv_kernel = cfg.final_conv_kernel
        self.with_se_cat = cfg.with_se_cat
        self.relu_inplace = cfg.relu_inplace
        self.in_index = cfg.in_index
        last_inp_channels = 18+36+72+144

        decoder_module = []
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm(last_inp_channels),
            nn.ReLU(inplace=self.relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=9,
                kernel_size=self.final_conv_kernel,
                stride=1,
                padding=1 if self.final_conv_kernel == 3 else 0))
        decoder_module.append('last_layer')

        if self.with_se_cat:
            self.se_cat = SELayer(last_inp_channels, reduction=16)
            decoder_module.append('se_cat')

        self.__setattr__('decoder', decoder_module)
        self.up = None

    def forward(self, backbone_out, size):
        backbone_out = [backbone_out[i] for i in self.in_index]

        upsampled_inputs = [
            F.interpolate(
                input=x,
                size=backbone_out[0].shape[2:],
                mode='bilinear',
                align_corners=False) for x in backbone_out
        ]
            
        backbone_out = torch.cat(upsampled_inputs, dim=1)
        
        x = self.last_layer(backbone_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=False)
        return x
