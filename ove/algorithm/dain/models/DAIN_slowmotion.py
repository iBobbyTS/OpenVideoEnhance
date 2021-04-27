import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks import (
    S2DF_3dense,
    MultipleBasicBlock_4,
    HourGlass,
    pwc_dc_net,
    DepthFlowProjectionModule,
    FilterInterpolationModule,
    FlowProjectionModule
)
from ..utils import Stack
from ove.utils.modeling import Sequential


class DAIN_slowmotion(torch.nn.Module):
    def __init__(
        self,
        size,
        padding,
        timestep=0.5,
        useAnimationMethod=0,
        rectify=False
    ):
        super().__init__()
        self.useAnimationMethod = useAnimationMethod
        self.rectify = rectify
        self.padding = padding
        self.time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]
        self.initScaleNets_filter, self.initScaleNets_filter1, self.initScaleNets_filter2 = self.get_MonoNet5(3, 16)
        self.ctxNet = S2DF_3dense()
        self.rectifyNet = MultipleBasicBlock_4() if rectify else lambda: None
        self._initialize_weights()
        self.flownets = pwc_dc_net(size)
        self.depthNet = HourGlass()
        self.filterModule = FilterInterpolationModule().cuda()

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                count += 1
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(
        self, cur_input_0, cur_input_2, target, count
    ):
        target[[count]] = cur_input_0[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]]
        count += 1
        device = torch.cuda.current_device()
        s1 = torch.cuda.Stream(device=device, priority=5)
        s2 = torch.cuda.Stream(device=device, priority=10)
        # STEP 3.2: concatenating the inputs.
        cur_offset_input = torch.cat((cur_input_0, cur_input_2), dim=1)
        cur_filter_input = cur_offset_input  # torch.cat((cur_input_0, cur_input_2), dim=1)
        # STEP 3.3: perform the estimation by the Three subpath Network
        with torch.cuda.stream(s1):
            if self.useAnimationMethod == 1 or self.useAnimationMethod == 2:
                temp = torch.cat((cur_filter_input[:, :3, ...], cur_filter_input[:, 3:, ...]), dim=0)
                temp = temp[:, 1:2, :, :]
            else:
                temp = self.depthNet(torch.cat((cur_filter_input[:, :3, ...], cur_filter_input[:, 3:, ...]), dim=0))
            log_depth = [temp[:cur_filter_input.size(0)], temp[cur_filter_input.size(0):]]

            if self.useAnimationMethod == 1:
                log_depth = [(d * 0) for d in log_depth]
            elif self.useAnimationMethod == 2:
                log_depth = [d for d in log_depth]
            else:
                log_depth = [temp[:cur_filter_input.size(0)], temp[cur_filter_input.size(0):]]

            cur_ctx_output = [
                torch.cat((
                    self.ctxNet(cur_filter_input[:, :3, ...]),
                    log_depth[0].detach()
                ), dim=1),
                torch.cat((
                    self.ctxNet(cur_filter_input[:, 3:, ...]),
                    log_depth[1].detach()
                ), dim=1)
            ]
            temp = self.forward_singlePath(
                self.initScaleNets_filter, cur_filter_input, 'filter'
            )
            cur_filter_output = [
                self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)
            ]

            if self.useAnimationMethod == 1:
                depth_inv = [(d * 0) + 1e-6 + 10000 for d in log_depth]
            else:
                depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

        with torch.cuda.stream(s2):
            cur_offset_outputs = [
                self.forward_flownets(self.flownets, cur_input_0, cur_input_2),
                self.forward_flownets(self.flownets, cur_input_2, cur_input_0)
            ]
        torch.cuda.synchronize()
        cur_offset_outputs = [
            self.FlowProject(cur_offset_outputs[0], depth_inv[0]),
            self.FlowProject(cur_offset_outputs[1], depth_inv[1])
        ]
        for temp_0, temp_1, timeoffset in zip(cur_offset_outputs[0], cur_offset_outputs[1], self.time_offsets):
            cur_offset_output = [temp_0, temp_1]
            ctx0, ctx2 = self.FilterInterpolate_ctx(
                self.filterModule,
                cur_ctx_output[0],
                cur_ctx_output[1],
                cur_offset_output,
                cur_filter_output
            )
            cur_output_temp, ref0, ref2 = self.FilterInterpolate(
                self.filterModule,
                cur_input_0,
                cur_input_2,
                cur_offset_output,
                cur_filter_output,
                self.filter_size ** 2,
                timeoffset
            )
            result = cur_output_temp[:, :, self.padding[2]: self.padding[3], self.padding[0]: self.padding[1]]
            if self.rectify:
                ref0 = ref0[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]]
                ref2 = ref2[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]]
                ctx0 = ctx0[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]]
                ctx2 = ctx2[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]]
                rectify_input = torch.cat((
                    result,
                    ref0,
                    ref2,
                    temp_0[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]],
                    temp_1[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]],
                    temp_0[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]],
                    temp_1[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]],
                    ctx0,
                    ctx2
                ), dim=1)
                result = self.rectifyNet(rectify_input) + result
            target[[count]] = result
            count += 1
        return count

    def forward_flownets(self, model, im1, im2):
        temp = model(im1, im2)
        temps = [20 * temp * time_offset for time_offset in self.time_offsets]
        temps = [F.interpolate(temp, scale_factor=4, mode='bilinear', align_corners=True) for temp in temps]
        return temps

    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()
        k = 0
        temp = []
        for layers in modulelist:
            if k == 0:
                temp = layers(input)
            else:
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers, nn.MaxPool2d):
                    stack.push(temp)
                temp = layers(temp)
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp, stack.pop()), dim=1)
                    else:
                        temp += stack.pop()
            k += 1
        return temp

    def get_MonoNet5(self, channel_in, channel_out):
        return (
            Sequential(
                self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1)),
                self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2)),
                self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2)),
                self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2)),
                self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2)),
                self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2)),
                self.conv_relu(512, 512, (3, 3), (1, 1)),
                self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2),
                self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2),
                self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2),
                self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2),
                self.conv_relu_unpool(32, 16, (3, 3), (1, 1), 2)
            ),
            self.conv_relu_conv(16, channel_out, (3, 3), (1, 1)),
            self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))
        )

    @staticmethod
    def FlowProject(inputs, depth=None):
        if depth is not None:
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input, depth) for input in inputs]
        else:
            outputs = [FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        return outputs

    @staticmethod
    def FilterInterpolate_ctx(filterModule, ctx0, ctx2, offset, filter):
        ctx0_offset = filterModule(ctx0, offset[0].detach(), filter[0].detach())
        ctx2_offset = filterModule(ctx2, offset[1].detach(), filter[1].detach())
        return ctx0_offset, ctx2_offset

    @staticmethod
    def FilterInterpolate(filterModule, ref0, ref2, offset, filter, filter_size2, time_offset):
        ref0_offset = filterModule(ref0, offset[0], filter[0])
        ref2_offset = filterModule(ref2, offset[1], filter[1])
        return ref0_offset * (1.0 - time_offset) + ref2_offset * (time_offset), ref0_offset, ref2_offset

    @staticmethod
    def conv_relu_conv(
        input_filter, output_filter, kernel_size, padding
    ):
        return Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, (1, 1), padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, (1, 1), padding),
        )

    @staticmethod
    def conv_relu(
        input_filter, output_filter, kernel_size, padding
    ):
        return Sequential(
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def conv_relu_maxpool(
        input_filter, output_filter, kernel_size,
        padding, kernel_size_pooling
    ):
        return Sequential(
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size_pooling)
        )

    @staticmethod
    def conv_relu_unpool(
        input_filter, output_filter, kernel_size,
        padding, unpooling_factor
    ):
        return Sequential(
            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, stride=(1,), padding=padding),
            nn.ReLU(inplace=True)
        )
