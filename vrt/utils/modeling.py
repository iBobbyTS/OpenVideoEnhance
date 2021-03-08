import math

import numpy
import cv2
import torch
from torch.nn import functional


def resize_hotfix(img):
    h, w = img.shape[2:4]
    img = torch.nn.functional.interpolate(
        img,
        size=(h + 2, w + 2),
        mode='bilinear', align_corners=True
    )
    return img[1:h + 1, 1:w + 1]


def resize_hotfix_numpy(img):
    h, w = img.shape[1:3]
    new = numpy.empty_like(img)
    for i, im in enumerate(img):
        im = cv2.resize(im, (w + 2, h + 2))
        new[i] = im[1:h + 1, 1:w + 1]
    return new


def cal_split(h_w, max_side_length):
    splition = []
    for i in range(math.ceil(h_w[0] / max_side_length)):
        for j in range(math.ceil(h_w[1] / max_side_length)):
            ws = i * max_side_length
            we = (i + 1) * max_side_length if (i + 1) * max_side_length <= h_w[0] else h_w[0]
            hs = j * max_side_length
            he = (j + 1) * max_side_length if (j + 1) * max_side_length <= h_w[1] else h_w[1]
            splition.append([ws, we, hs, he])
    return splition


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
        self.padding_result = (left, right, top, bottom)
        self.slice = (left, width + left, top, height + top)
        self.padded_size = (width + left + right, height + top + bottom)
        self.pader = {
            'replication': torch.nn.ReplicationPad2d,
            'reflection': torch.nn.ReflectionPad2d,
            'constant': torch.nn.ConstantPad3d,
        }[extend_func]([left, right, top, bottom], *args)

    def pad(self, tensor):
        return self.pader(tensor)
