import torch
from torch.autograd import Function
from torch.nn import Module
import ove_ext.dain.filter_interpolation as my_lib
from ove.utils.modeling import make_contiguous


class FilterInterpolationLayer(Function):
    @staticmethod
    def forward(ctx, input1, input2, input3):
        input1, input2, input3 = make_contiguous((input1, input2, input3))
        output = torch.zeros_like(input1)
        my_lib.forward(input1, input2, input3, output)
        return output

    @staticmethod
    def backward(*args, **kwargs):
        return


class FilterInterpolationModule(Module):
    @staticmethod
    def forward(input1, input2, input3):
        return FilterInterpolationLayer.apply(input1, input2, input3)
