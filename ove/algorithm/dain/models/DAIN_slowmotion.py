import torch
import torch.nn as nn

from ove.utils.modeling import Sequential
from ..networks import (
    FilterInterpolationModule, DepthFlowProjectionModule,
    MultipleBasicBlock, S2DF, PWCDCNet, HourGlass
)
from ..utils import Stack


class DAIN_slowmotion(nn.Module):
    def __init__(
            self,
            size,
            batch_size=1,
            sf=2,
            rectify=False,
            padding=None,
            useAnimationMethod=0
    ):
        super().__init__()
        self.rectify = rectify
        self.padding = padding
        self.batch_size = batch_size
        self.useAnimationMethod = useAnimationMethod
        self.time_offsets = [kk / sf for kk in range(1, sf)]
        self.initScaleNets_filter, self.initScaleNets_filter1, self.initScaleNets_filter2 = self.get_MonoNet5()
        self.ctxNet = S2DF() if rectify else None
        self.rectifyNet = MultipleBasicBlock() if rectify else None
        self._initialize_weights()
        self.flownets = PWCDCNet(*size)
        self.depthNet = HourGlass
        self.filterModule = FilterInterpolationModule().cuda()
        self.depth_flow_projection = DepthFlowProjectionModule()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, I0, I1, target, count):
        target[[count]] = I0[:, :, self.padding[2]:self.padding[3], self.padding[0]:self.padding[1]]
        count += 1

        cat0 = torch.cat((I0, I1), dim=0)
        cat1 = torch.cat((I0, I1), dim=1)
        with torch.cuda.stream(torch.cuda.current_stream()):
            if self.useAnimationMethod:
                temp = I1[:, 1:2, :, :]
            else:
                temp = self.depthNet(cat0)

            log_depth = [temp[:self.batch_size], temp[self.batch_size:]]
            if self.useAnimationMethod == 1:
                log_depth = [(d * 0) for d in log_depth]

            temp = self.forward_singlePath(self.initScaleNets_filter, cat1)
            cur_filter_output = [
                self.forward_singlePath(self.initScaleNets_filter1, temp),
                self.forward_singlePath(self.initScaleNets_filter2, temp)
            ]
            if self.useAnimationMethod == 1:
                depth_inv = [(d * 0) + 1e-6 + 10000 for d in log_depth]
            else:
                depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

        with torch.cuda.stream(torch.cuda.current_stream()):
            cur_offset_outputs = [
                self.forward_flownets(I0, I1, inv=False),
                self.forward_flownets(I1, I0, inv=True)
            ]

        torch.cuda.synchronize()  # synchronize s1 and s2

        cur_offset_outputs = [
            self.FlowProject(cur_offset_outputs[0], depth_inv[0]),
            self.FlowProject(cur_offset_outputs[1], depth_inv[1])
        ]

        for temp_0, temp_1, timeoffset in zip(cur_offset_outputs[0], cur_offset_outputs[1], self.time_offsets):
            cur_offset_output = [temp_0, temp_1]
            cur_output_temp, ref0, ref2 = self.FilterInterpolate(
                I0, I1,
                cur_offset_output, cur_filter_output,
                timeoffset
            )
            cur_output_temp = cur_output_temp[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
            if self.rectify:
                cur_ctx_output = [
                    torch.cat((self.ctxNet(I0), log_depth[0].detach()), dim=1),
                    torch.cat((self.ctxNet(I1), log_depth[1].detach()), dim=1)
                ]
                ctx0, ctx2 = self.FilterInterpolate_ctx(
                    cur_ctx_output[0], cur_ctx_output[1],
                    cur_offset_output, cur_filter_output, timeoffset
                )
                ref0 = ref0[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                ref2 = ref2[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                ctx0 = ctx0[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                ctx2 = ctx2[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                rectify_input = torch.cat((
                    cur_output_temp, ref0, ref2,
                    temp_0[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]],
                    temp_1[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]],
                    cur_filter_output[0][:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]],
                    cur_filter_output[1][:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]],
                    ctx0, ctx2
                ), dim=1)
                cur_output_temp = self.rectifyNet(rectify_input) + cur_output_temp
            target[[count]] = cur_output_temp
            count += 1
        return count

    def forward_flownets(self, im1, im2, inv):
        temp = self.flownets(im1, im2)
        temps = [20.0 * temp * time_offset for time_offset in self.time_offsets]
        if inv:
            temps = temps[::-1]
        temps = [nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(temp) for temp in temps]
        return temps

    def forward_singlePath(self, model, input):
        stack = Stack()
        k = 0
        temp = []
        for layers in model:
            if k == 0:
                temp = layers(input)
            else:
                if isinstance(layers, (nn.AvgPool2d, nn.MaxPool2d)):
                    stack.push(temp)
                temp = layers(temp)
                if isinstance(layers, nn.Upsample):
                    temp += stack.pop()
            k += 1
        return temp

    def get_MonoNet5(self):
        model = Sequential(
            *self.conv_relu(6, 16, (3, 3), (1, 1)),
            *self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2)),
            *self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2)),
            *self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2)),
            *self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2)),
            *self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2)),
            *self.conv_relu(512, 512, (3, 3), (1, 1)),
            *self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2),
            *self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2),
            *self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2),
            *self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2),
            *self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)
        )
        branch1 = self.conv_relu_conv(16, 16, (3, 3), (1, 1))
        branch2 = self.conv_relu_conv(16, 16, (3, 3), (1, 1))

        return model, branch1, branch2

    def FlowProject(self, inputs, depth=None):
        return [self.depth_flow_projection(x, depth) for x in inputs]

    def FilterInterpolate_ctx(self, ctx0, ctx2, offset, filter, timeoffset):
        ctx0_offset = self.filterModule(ctx0, offset[0].detach(), filter[0].detach())
        ctx2_offset = self.filterModule(ctx2, offset[1].detach(), filter[1].detach())
        return ctx0_offset, ctx2_offset

    def FilterInterpolate(self, ref0, ref2, offset, filter, time_offset):
        ref0_offset = self.filterModule(ref0, offset[0], filter[0])
        ref2_offset = self.filterModule(ref2, offset[1], filter[1])
        return ref0_offset * (1.0 - time_offset) + ref2_offset * time_offset, ref0_offset, ref2_offset

    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size, padding):
        layers = Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
        )
        return layers

    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size, padding):
        layers = [
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True)
        ]
        return layers

    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                          padding, kernel_size_pooling):
        layers = [
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size_pooling)
        ]
        return layers

    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size, padding, unpooling_factor):
        layers = [
            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
        ]
        return layers
