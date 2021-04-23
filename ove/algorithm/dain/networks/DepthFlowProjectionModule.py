import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import ove_ext.dain.depth_flow_projection as my_lib


class DepthFlowProjectionLayer(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input1, input2, requires_grad):
        assert (input1.is_contiguous())
        assert (input2.is_contiguous())
        fillhole = 0 if requires_grad else 1
        count = torch.cuda.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
        output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        err = my_lib.DepthFlowProjectionLayer_gpu_forward(input1, input2, count, output, fillhole)
        if err != 0:
            print(err)
        ctx.save_for_backward(input1, input2, count, output)
        ctx.fillhole = fillhole
        return output

    @staticmethod
    def backward(*args, **kwargs):
        pass


class DepthFlowProjectionModule(Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.requires_grad = requires_grad

    def forward(self, input1, input2):
        return DepthFlowProjectionLayer.apply(input1, input2, self.requires_grad)
