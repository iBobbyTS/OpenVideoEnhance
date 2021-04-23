import torch
from torch.nn import Module
from torch.autograd import Function
import ove_ext.dain.filter_interpolation as my_lib
from ove.utils.io import empty_cache


class FilterInterpolationLayer(Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input1, input2, input3):
        empty_cache()
        output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        my_lib.FilterInterpolationLayer_gpu_forward(input1, input2, input3, output)
        empty_cache()
        ctx.save_for_backward(input1, input2, input3)
        return output

    @staticmethod
    def backward(*args, **kwargs):
        pass


class FilterInterpolationModule(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input1, input2, input3):
        return FilterInterpolationLayer.apply(input1, input2, input3)
