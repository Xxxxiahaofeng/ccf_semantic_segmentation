import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.utils import ConvBNReLU, SeparableConvBNReLU, ASPPModule


class DeepLabV3P(nn.Layer):
    """
    The DeepLabV3Plus implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = DeepLabV3PHead(num_classes, backbone_indices,
                                   backbone_channels, aspp_ratios,
                                   aspp_out_channels)

        self.pretrained = pretrained

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(logit, x.shape[2:], mode='bilinear')
            for logit in logit_list
        ]


class DeepLabV3PHead(nn.Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels):
        super().__init__()

        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            use_sep_conv=True,
            image_pooling=True)
        self.decoder = Decoder(num_classes, backbone_channels[0])
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)

        return logit_list


class Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
    """

    def __init__(self, num_classes, in_channels):
        super(Decoder, self).__init__()

        self.conv_bn_relu1 = ConvBNReLU(
            in_channels=in_channels, out_channels=48, kernel_size=1)

        self.conv_bn_relu2 = SeparableConvBNReLU(
            in_channels=304, out_channels=256, kernel_size=3, padding=1)
        self.conv_bn_relu3 = SeparableConvBNReLU(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv = nn.Conv2D(
            in_channels=256, out_channels=num_classes, kernel_size=1)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        x = F.interpolate(x, low_level_feat.shape[2:], mode='bilinear')
        x = paddle.concat([x, low_level_feat], axis=1)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x