"""
@author: Yi-Wei Chen <feberium@gmail.com>
"""
import os

import torch.nn as nn
import torch

import sys
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai8x-training"))

import ai8x
from ai8x import FusedConv2dBNReLU, FusedMaxPoolConv2dBNReLU, Conv2d


class TinyYoloV2(nn.Module):
    def __init__(self, num_classes,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]):
        super(TinyYoloV2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        self.stage1_conv1 = FusedConv2dBNReLU(in_channels=3, out_channels=32, kernel_size=3, stride=1,
                                              padding=1, bias=False)
        self.stage1_conv2 = FusedMaxPoolConv2dBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                                                     padding=1, bias=False, pool_size=2, pool_stride=2)
        self.stage1_conv3 = FusedMaxPoolConv2dBNReLU(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                                     padding=1, bias=False, pool_size=2, pool_stride=2)
        self.stage1_conv4 = FusedConv2dBNReLU(in_channels=128, out_channels=64, kernel_size=1, stride=1,
                                              padding=0, bias=False)
        self.stage1_conv5 = FusedConv2dBNReLU(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                              padding=1, bias=False)
        self.stage1_conv6 = FusedMaxPoolConv2dBNReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                                     padding=1, bias=False, pool_size=2, pool_stride=2)
        self.stage1_conv7 = FusedConv2dBNReLU(in_channels=256, out_channels=128, kernel_size=1, stride=1,
                                              padding=0, bias=False)
        self.stage1_conv8 = FusedConv2dBNReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                              padding=1, bias=False)
        self.stage1_conv9 = FusedMaxPoolConv2dBNReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                                     padding=1, bias=False, pool_size=2, pool_stride=2)
        self.stage1_conv10 = FusedConv2dBNReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1,
                                               padding=0, bias=False)
        self.stage1_conv11 = FusedConv2dBNReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                               padding=1, bias=False)
        self.stage1_conv12 = FusedConv2dBNReLU(in_channels=512, out_channels=256, kernel_size=1, stride=1,
                                               padding=0, bias=False)
        self.stage1_conv13 = FusedConv2dBNReLU(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                               padding=1, bias=False)

        self.stage2_a_conv1 = FusedMaxPoolConv2dBNReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1,
                                                       padding=1, bias=False, pool_size=2, pool_stride=2)
        self.stage2_a_conv2 = FusedConv2dBNReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1,
                                                padding=0, bias=False)
        self.stage2_a_conv3 = FusedConv2dBNReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1,
                                                padding=1, bias=False)
        self.stage2_a_conv4 = FusedConv2dBNReLU(in_channels=1024, out_channels=512, kernel_size=1, stride=1,
                                                padding=0, bias=False)
        self.stage2_a_conv5 = FusedConv2dBNReLU(in_channels=512, out_channels=1024, kernel_size=3, stride=1,
                                                padding=1, bias=False)
        self.stage2_a_conv6 = FusedConv2dBNReLU(in_channels=1024, out_channels=1024, kernel_size=3, stride=1,
                                                padding=1, bias=False)
        self.stage2_a_conv7 = FusedConv2dBNReLU(in_channels=1024, out_channels=1024, kernel_size=3, stride=1,
                                                padding=1, bias=False)

        self.stage2_b_conv = FusedConv2dBNReLU(in_channels=512, out_channels=64, kernel_size=1, stride=1,
                                               padding=0, bias=False)

        self.stage3_conv1 = FusedConv2dBNReLU(in_channels=256 + 1024, out_channels=1024, kernel_size=3, stride=1,
                                              padding=1, bias=False)
        self.stage3_conv2 = Conv2d(in_channels=1024, out_channels=len(self.anchors) * (5 + num_classes), kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        # output_1 = self.stage2_a_maxpl(output)
        output_1 = output
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)

        return output


if __name__ == "__main__":
    ai8x.set_device(device=85, simulate=False, round_avg=False)
    net = TinyYoloV2(20)
    print(net.stage1_conv1)
