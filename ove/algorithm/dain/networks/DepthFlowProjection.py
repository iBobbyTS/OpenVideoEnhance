import torch
from torch.autograd import Function
from torch.nn import Module
import ove_ext.dain.depth_flow_projection as my_lib


class DepthFlowProjectionLayer(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        output = torch.zeros_like(input1)
        count = torch.zeros((input1.size(0), 1, input1.size(2), input1.size(3)), device='cuda')
        err = my_lib.forward(input1, input2, count, output, 1)
        if err != 0:
            print(err)
        return output

    @staticmethod
    def backward(*args, **kwargs):
        pass


class DepthFlowProjectionModule(Module):
    def forward(self, input1, input2):
        return DepthFlowProjectionLayer.apply(input1, input2)
