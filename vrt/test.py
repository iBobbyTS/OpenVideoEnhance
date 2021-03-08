from time import time
import numpy
import torch
import cv2
from vrt.utils.tensor import Tensor
x = numpy.array(cv2.imread('/Users/ibobby/Pictures/New Folder With Items 2/who_wins.tiff'))
x = Tensor(
	tensor=x, shape_order='hwc', channel_order='bgr', range_=(0, 255)
)
# x.convert(place='torch')
# x.convert(
# 	place='numpy', channel_order='rgb', shape_order='chw', range_=(0, 1), dtype='float32'
# )
# x.convert(
# 	place='torch', channel_order='bgr', shape_order='hwc', range_=(0, 255), dtype='uint8'
# )
# x.convert(place='numpy', channel_order='rgb')
# cv2.imwrite('/Users/ibobby/Pictures/New Folder With Items 2/who_wins2.tiff', x.tensor)
print(x)