import torch
from torch.autograd import Function
from torch.nn import Module
import ove_ext.dain.flow_projection as my_lib
from ove.utils.modeling import make_contiguous


class FlowProjectionLayer(Function):
    def forward(self, input1, requires_grad=False):
        input1 = make_contiguous(input1)
        fillhole = 0 if requires_grad else 1
        output = torch.zeros_like(input1)
        count = torch.zeros((input1.size(0), 1, input1.size(2), input1.size(3)), device='cuda')
        err = my_lib.forward(input1, count, output, fillhole)
        if err != 0:
            print(err)
        return output

    @staticmethod
    def backward(*args, **kwargs):
        return


class FlowProjectionModule(Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.f = FlowProjectionLayer()
        self.requires_grad = requires_grad

    def forward(self, input1):
        return self.f(input1, self.requires_grad)
