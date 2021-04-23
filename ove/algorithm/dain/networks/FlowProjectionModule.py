import torch
from torch.nn import Module
from torch.autograd import Function
import ove_ext.dain.flow_projection as my_lib


class FlowProjectionLayer(Function):
    def __init__(self, requires_grad):
        super().__init__()
        self.requires_grad = requires_grad

    @staticmethod
    def forward(ctx, input1, requires_grad):
        fillhole = 0 if requires_grad else 1
        count = torch.cuda.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
        output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        err = my_lib.FlowProjectionLayer_gpu_forward(input1, count, output, fillhole)
        if err != 0:
            print(err)
        ctx.save_for_backward(input1, count)
        ctx.fillhole = fillhole
        return output

    @staticmethod
    def backward(*args, **kwargs):
        pass


class FlowProjectionModule(Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.f = FlowProjectionLayer(requires_grad)

    def forward(self, input1):
        return self.f(input1)
