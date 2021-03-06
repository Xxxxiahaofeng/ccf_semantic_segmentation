# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.utils import ConvBN, ConvBNReLU
from models.utils import constant_init, normal_init


class UNet(nn.Layer):
    """
    The UNet implementation based on PaddlePaddle.
    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).
    Args:
        num_classes (int): The unique number of target classes.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self, num_classes, use_deconv=False):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder(use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        return logit



class Encoder(nn.Layer):
    def __init__(self):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBNReLU(3, 64, 3), ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 1024]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts

class Decoder(nn.Layer):
    def __init__(self, use_deconv=False):
        super().__init__()

        up_channels = [[1024, 512], [512, 256], [256, 128], [128, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1])
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.deconv = nn.Conv2DTranspose(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
            padding=0)
        # in_channels = in_channels + out_channels

        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):

        x = self.deconv(x)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x