# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ..my_package.FilterInterpolation import FilterInterpolationModule
from ..my_package.FlowProjection import FlowProjectionModule  # ,FlowFillholeModule
from ..my_package.DepthFlowProjection import DepthFlowProjectionModule
from ..Stack import Stack
from ..PWCNet import pwc_dc_net
from ..S2D_models import S2DF_3dense
from ..Resblock import MultipleBasicBlock_4
from ..MegaDepth import HourGlass


class DAIN_slowmotion(torch.nn.Module):
    def __init__(self,
                 padding,
                 channel=3,
                 filter_size=4,
                 timestep=0.5,
                 training=True,
                 useAnimationMethod=0,
                 rectify=False):

        # base class initialization
        super(DAIN_slowmotion, self).__init__()

        self.filter_size = filter_size
        self.training = training
        self.useAnimationMethod = useAnimationMethod
        self.rectify = rectify
        self.padding = padding

        self.timestep = timestep
        self.numFrames = int(1.0 / timestep) - 1

        i = 0
        self.initScaleNets_filter, self.initScaleNets_filter1, self.initScaleNets_filter2 = \
            self.get_MonoNet5(channel if i == 0 else channel + filter_size * filter_size, filter_size * filter_size,
                              "filter")

        self.ctxNet = S2DF_3dense()
        self.ctx_ch = 3 * 64 + 3

        self.rectifyNet = MultipleBasicBlock_4(3 + 3 + 3 + 2 * 1 + 2 * 2 + 16 * 2 + 2 * self.ctx_ch,
                                               128) if rectify else lambda: None

        self._initialize_weights()

        if self.training:
            self.flownets = pwc_dc_net("PWCNet/pwc_net.pth.tar")
        else:
            self.flownets = pwc_dc_net()
        self.div_flow = 20.0

        # extract depth information
        if self.training:
            self.depthNet = HourGlass("MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")
        else:
            self.depthNet = HourGlass()
        self.filterModule = FilterInterpolationModule.interpolation().cuda()

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

    def forward(self, input_0, input_2):

        """
        Parameters
        ----------
        inputs：shape (3, batch, 3, width, height)
        -----------
        """
        losses = []
        offsets = []
        filters = []
        occlusions = []

        device = torch.cuda.current_device()
        # s1 = torch.cuda.Stream(device=device, priority=5)
        # s2 = torch.cuda.Stream(device=device, priority=10) #PWC-Net is slow, need to have higher priority
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()

        '''
            STEP 1: sequeeze the input 
        '''

        alpha = False
        if input_0.size(1) == 4:
            # print("There is an alpha channel")
            input_0_alpha = input_0[:, 3:4, :, :].cuda()
            input_0 = input_0[:, 0:3, :, :].cuda()
            input_2_alpha = input_2[:, 3:4, :, :].cuda()
            input_2 = input_2[:, 0:3, :, :].cuda()
            alpha = True
        elif input_0.size(1) == 3:
            # print("Normal 3 channel image, such as RGB")
            input_0 = input_0.cuda()
            input_2 = input_2.cuda()
        else:
            print("Bug??")

        # prepare the input data of current scale
        cur_input_0 = input_0
        if self.training == True:
            pass
            # cur_input_1 = input_1
        cur_input_2 = input_2

        '''
            STEP 3.2: concatenating the inputs.
        '''
        cur_offset_input = torch.cat((cur_input_0, cur_input_2), dim=1)
        cur_filter_input = cur_offset_input  # torch.cat((cur_input_0, cur_input_2), dim=1)

        '''
            STEP 3.3: perform the estimation by the Three subpath Network 
        '''
        time_offsets = [kk * self.timestep for kk in range(1, 1 + self.numFrames, 1)]

        with torch.cuda.stream(s1):
            # temp  = self.depthNet(torch.cat((cur_filter_input[:, :3, ...],
            #                                 cur_filter_input[:, 3:, ...]),dim=0))

            if self.useAnimationMethod == 1 or self.useAnimationMethod == 2:
                temp = torch.cat((cur_filter_input[:, :3, ...], cur_filter_input[:, 3:, ...]), dim=0)
                temp = temp[:, 1:2, :, :]
            else:
                temp = self.depthNet(torch.cat((cur_filter_input[:, :3, ...], cur_filter_input[:, 3:, ...]), dim=0))

            log_depth = [temp[:cur_filter_input.size(0)], temp[cur_filter_input.size(0):]]

            if self.useAnimationMethod == 1:
                log_depth = [(d * 0) for d in log_depth]
            if self.useAnimationMethod == 2:
                log_depth = [d for d in log_depth]

            cur_ctx_output = [
                torch.cat((self.ctxNet(cur_filter_input[:, :3, ...]),
                           log_depth[0].detach()), dim=1),
                torch.cat((self.ctxNet(cur_filter_input[:, 3:, ...]),
                           log_depth[1].detach()), dim=1)
            ]
            temp = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input, 'filter')
            cur_filter_output = [self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                                 self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)]

            # depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]
            if self.useAnimationMethod == 1:
                depth_inv = [(d * 0) + 1e-6 + 10000 for d in log_depth]
            else:
                depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

        with torch.cuda.stream(s2):
            cur_offset_outputs = [
                self.forward_flownets(self.flownets, cur_input_0, cur_input_2, time_offsets),
                self.forward_flownets(self.flownets, cur_input_2, cur_input_0, time_offsets[::-1])
            ]

        torch.cuda.synchronize()  # synchronize s1 and s2

        cur_offset_outputs = [
            self.FlowProject(cur_offset_outputs[0], depth_inv[0]),
            self.FlowProject(cur_offset_outputs[1], depth_inv[1])
        ]

        '''
            STEP 3.4: perform the frame interpolation process 
        '''
        cur_output_rectified = []
        cur_output = []

        for temp_0, temp_1, timeoffset in zip(cur_offset_outputs[0], cur_offset_outputs[1], time_offsets):
            cur_offset_output = [temp_0, temp_1]  # [cur_offset_outputs[0][0], cur_offset_outputs[1][0]]
            ctx0, ctx2 = self.FilterInterpolate_ctx(self.filterModule, cur_ctx_output[0], cur_ctx_output[1],
                                                    cur_offset_output, cur_filter_output, timeoffset)

            cur_output_temp, ref0, ref2 = self.FilterInterpolate(self.filterModule, cur_input_0, cur_input_2,
                                                                 cur_offset_output,
                                                                 cur_filter_output, self.filter_size ** 2, timeoffset)

            cur_output_temp = cur_output_temp[:, :, self.padding[2]: self.padding[3], self.padding[0]: self.padding[1]]

            cur_output.append(cur_output_temp[0])

            if self.rectify:
                ref0 = ref0[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                ref2 = ref2[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                ctx0 = ctx0[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                ctx2 = ctx2[:, :, self.padding[2]:self.padding[3], self.padding[0]: self.padding[1]]
                rectify_input = torch.cat((cur_output_temp, ref0, ref2,
                                           cur_offset_output[0][:, :, self.padding[2]:self.padding[3],
                                           self.padding[0]: self.padding[1]],
                                           cur_offset_output[1][:, :, self.padding[2]:self.padding[3],
                                           self.padding[0]: self.padding[1]],
                                           cur_filter_output[0][:, :, self.padding[2]:self.padding[3],
                                           self.padding[0]: self.padding[1]],
                                           cur_filter_output[1][:, :, self.padding[2]:self.padding[3],
                                           self.padding[0]: self.padding[1]],
                                           ctx0, ctx2
                                           ), dim=1)
                cur_output_rectified_temp = self.rectifyNet(rectify_input) + cur_output_temp
                cur_output_rectified.append(cur_output_rectified_temp[0])

        '''
            STEP 3.5: for training phase, we collect the variables to be penalized.
        '''
        if self.training:
            # losses += [cur_output - cur_input_1]
            # losses += [cur_output_rectified - cur_input_1]
            offsets += [cur_offset_output]
            filters += [cur_filter_output]
        '''
            STEP 4: return the results
        '''
        if self.training:
            # if in the training phase, we output the losses to be minimized.
            # return losses, loss_occlusion
            return losses, offsets, filters, occlusions
        else:
            # cur_outputs = [cur_output,cur_output_rectified]
            # return cur_outputs,cur_offset_output,cur_filter_output
            if self.rectify:
                return cur_output_rectified
            else:
                return cur_output

    def forward_flownets(self, model, im1, im2, time_offsets):

        if time_offsets is None:
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        temp = model(im1, im2)  # this is a single direction motion results, but not a bidirectional one

        temps = [self.div_flow * temp * time_offset for time_offset in
                 time_offsets]  # single direction to bidirection should haven it.
        temps = [nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(temp) for temp in
                 temps]  # nearest interpolation won't be better i think
        return temps

    '''keep this function'''

    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:  # self.initScaleNets_offset:
            # print(type(layers).__name__)
            # print(k)
            # if k == 27:
            #     print(k)
            #     pass
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers, nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp, stack.pop()),
                                         dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp

    '''keep this funtion'''

    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32, 16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))

        return (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    '''keep this function'''

    @staticmethod
    def FlowProject(inputs, depth=None):
        if depth is not None:
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input, depth) for input in inputs]
        else:
            outputs = [FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        return outputs

    '''keep this function'''

    @staticmethod
    def FilterInterpolate_ctx(filterModule, ctx0, ctx2, offset, filter, timeoffset):
        ##TODO: which way should I choose

        ctx0_offset = filterModule(ctx0, offset[0].detach(), filter[0].detach())
        ctx2_offset = filterModule(ctx2, offset[1].detach(), filter[1].detach())

        return ctx0_offset, ctx2_offset
        # ctx0_offset = FilterInterpolationModule()(ctx0.detach(), offset[0], filter[0])
        # ctx2_offset = FilterInterpolationModule()(ctx2.detach(), offset[1], filter[1])
        #
        # return ctx0_offset, ctx2_offset

    '''Keep this function'''

    @staticmethod
    def FilterInterpolate(filterModule, ref0, ref2, offset, filter, filter_size2, time_offset):
        ref0_offset = filterModule(ref0, offset[0], filter[0])
        ref2_offset = filterModule(ref2, offset[1], filter[1])

        # occlusion0, occlusion2 = torch.split(occlusion, 1, dim=1)
        # print((occlusion0[0,0,1,1] + occlusion2[0,0,1,1]))
        # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset) / (occlusion0 + occlusion2)
        # output = * ref0_offset + occlusion[1] * ref2_offset
        # automatically broadcasting the occlusion to the three channels of and image.
        # return output
        # return ref0_offset/2.0 + ref2_offset/2.0, ref0_offset,ref2_offset
        return ref0_offset * (1.0 - time_offset) + ref2_offset * (time_offset), ref0_offset, ref2_offset

    '''keep this function'''

    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                       padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
        )
        return layers

    '''keep this fucntion'''

    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                  padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

            nn.ReLU(inplace=True)
        ])
        return layers

    '''keep this function'''

    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                          padding, kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

            nn.ReLU(inplace=True),

            # nn.BatchNorm2d(output_filter),

            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers

    '''klkeep this function'''

    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                         padding, unpooling_factor):

        layers = nn.Sequential(*[

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear', align_corners=True),

            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

            nn.ReLU(inplace=True),

            # nn.BatchNorm2d(output_filter),

            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers
