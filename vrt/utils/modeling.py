import cv2
import torch


def resize_hotfix(img):
    h, w = img.shape[0:2]
    return cv2.resize(img, (w+2, h+2))[1:h+1, 1:w+1]


def calculate_expansion(width, height, factor, aline_to_edge):
    # Pader
    vertical_pad = factor - _ if (_ := height % factor) else 0
    horizontal_pad = factor - _ if (_ := width % factor) else 0
    if aline_to_edge:
        bottom = vertical_pad
        right = horizontal_pad
        top = 0
        left = 0
    else:
        left = horizontal_pad // 2
        right = horizontal_pad - left
        top = vertical_pad // 2
        bottom = vertical_pad - top
    return left, right, top, bottom


class Pader:
    def __init__(
            self,
            width, height, factor, aline_to_edge=False, extend_func='replication',
            *args, **kwargs
    ):
        left, right, top, bottom = calculate_expansion(width, height, factor, aline_to_edge)
        self.pading_result = (left, right, top, bottom)
        self.paded_size = (width + left + right, height + top + bottom)
        self.pader = {
            'replication': torch.nn.ReplicationPad2d,
            'reflection': torch.nn.ReflectionPad2d,
            'constant': torch.nn.ConstantPad3d,
        }[extend_func]([left, right, top, bottom], *args)

    def pad(self, tensor):
        return self.pader(tensor)
