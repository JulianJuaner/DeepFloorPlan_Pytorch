import torch
import torch.nn as nn
import torch.nn.functional as F
from sourcecode.model.utils import SELayer

def build_head(head_cfg):
    if head_cfg.type == 'HRHead':
        return FloorHead(head_cfg)
    elif head_cfg.type == 'ClassWareHead':
        return ClassWareHead(head_cfg)

BatchNorm = nn.BatchNorm2d

# Reference: GitHub: HRNet/HRNet-Semantic-Segmentation
class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x c x k
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.relu_inplace = True
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.key_channels),
            nn.ReLU(inplace=self.relu_inplace),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.key_channels),
            nn.ReLU(inplace=self.relu_inplace),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.key_channels),
            nn.ReLU(inplace=self.relu_inplace),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.key_channels),
            nn.ReLU(inplace=self.relu_inplace),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.key_channels),
            nn.ReLU(inplace=self.relu_inplace),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.in_channels),
            nn.ReLU(inplace=self.relu_inplace),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)

        return context

class ClassWareHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.final_conv_kernel = cfg.final_conv_kernel
        self.avg_pool = cfg.avg_pool
        self.relu_inplace = cfg.relu_inplace
        self.in_index = cfg.in_index
        self.context_extractor = SpatialGather_Module(9, 1)
        self.key_channels = cfg.key_channels
        last_inp_channels = 18+36+72+144

        decoder_module = []

        self.object_attention = ObjectAttentionBlock(last_inp_channels,
                                         self.key_channels)
        self.query_pix = self.transfer_function(last_inp_channels,
                                         last_inp_channels)
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
                padding=1 if self.final_conv_kernel == 3 else 0),
            )

        self.cls_head = nn.Sequential(
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
                padding=1 if self.final_conv_kernel == 3 else 0),
            )

        if self.avg_pool:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
            self.global_avg_conv = self.transfer_function(last_inp_channels,
                                         self.key_channels)
            self.concat_conv = self.transfer_function(last_inp_channels+self.key_channels,
                                         last_inp_channels)

        self.__setattr__('decoder', decoder_module)

    def transfer_function(self, in_chn, out_chn):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chn,
                out_channels=in_chn,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm(in_chn),
            nn.ReLU(inplace=self.relu_inplace),
            nn.Conv2d(
                in_channels=in_chn,
                out_channels=out_chn,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm(out_chn),
            nn.ReLU(inplace=self.relu_inplace))

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
        batch_size, c, h, w = backbone_out.size(0), backbone_out.size(1), \
                              backbone_out.size(2), backbone_out.size(3)
        if self.avg_pool:
            avg_pool = self.adaptive_pool(backbone_out)
            avg_pool = self.global_avg_conv(avg_pool)
            backbone_out = torch.cat([F.interpolate(
                                        avg_pool,
                                        backbone_out.size()[2:],
                                        mode='bilinear',
                                        align_corners=False
                                    ), backbone_out], dim=1)
            backbone_out = self.concat_conv(backbone_out)

        batch_size, c, h, w = backbone_out.size(0), backbone_out.size(1), \
                              backbone_out.size(2), backbone_out.size(3)

        aux = self.last_layer(backbone_out)
        aux_prob = F.softmax(aux, dim=1)

        pixel_feats = self.query_pix(backbone_out)

        contexts = self.context_extractor(pixel_feats, aux_prob)
        contexts = self.object_attention(pixel_feats, contexts)
        x = self.cls_head(contexts)

        # x = aux+res
        x = F.interpolate(x, size, mode='bilinear', align_corners=False)
        aux = F.interpolate(aux, size, mode='bilinear', align_corners=False)
        return [x, aux]
    
    
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
