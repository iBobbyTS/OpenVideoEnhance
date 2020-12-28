import warnings
import os

import numpy
import torch

from . import networks
from utils.io_utils import empty_cache


class rter:
    warnings.filterwarnings("ignore")
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    def __init__(self, height: int, width: int, batch_size=1,
                 model_directory='model_weights/DAIN/best.pth',
                 *args, **kwargs):
        # args
        self.batch_size = batch_size
        self.rectify = kwargs['rectify']

        # pader
        if width != ((width >> 7) << 7):
            intWidth_pad = (((width >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - width) / 2)
            intPaddingRight = intWidth_pad - width - intPaddingLeft
        else:
            intPaddingLeft = 32
            intPaddingRight = 32
        if height != ((height >> 7) << 7):
            intHeight_pad = (((height >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - height) / 2)
            intPaddingBottom = intHeight_pad - height - intPaddingTop
        else:
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])
        self.hs = intPaddingLeft  # Horizontal Start
        self.he = intPaddingLeft + width
        self.vs = intPaddingTop
        self.ve = intPaddingTop + height  # Vertical End

        # Model
        model = networks.__dict__[kwargs['net_name']](
            padding=(self.hs, self.he, self.vs, self.ve), channel=3, filter_size=4,
            timestep=1/kwargs['coef'], rectify=kwargs['rectify'], training=False).cuda()
        empty_cache()

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(model_directory).items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.model = model.eval()

        self.ndarray2tensor = lambda frame: pader(torch.unsqueeze((torch.cuda.ByteTensor(frame)[:, :, :3].permute(2, 0, 1).float() / 255), 0))
        self.batch = torch.cuda.FloatTensor(batch_size + 1, 3, intPaddingTop + height + intPaddingBottom, intPaddingLeft + width + intPaddingRight)

        if kwargs['net_name'] == 'DAIN_slowmotion':
            self.tensor2ndarray = lambda y_: [(255*item).clamp(0.0, 255.0).byte()[0, :, self.vs:self.ve,self.hs:self.he].permute(1, 2, 0).cpu().numpy() for item in y_]
        elif kwargs['net_name'] == 'DAIN':
            self.tensor2ndarray = lambda y_: [(255*item).clamp(0.0, 255.0).byte()[:, self.vs:self.ve,self.hs:self.he].permute(1, 2, 0).cpu().numpy() for item in y_]

    def init_batch(self, buffer):
        self.inited = False

    def store_ndarray_in_tensor(self, frame: numpy.ndarray, index: int):  # 内部调用
        self.batch[index] = self.ndarray2tensor(frame)

    def rt(self, frames, *args, **kwargs):
        if not self.inited:
            self.store_ndarray_in_tensor(frames[0], 0)
            self.inited = True
            return [frames[0]]

        for i, f in enumerate(frames, 1):
            self.store_ndarray_in_tensor(f, i)

        empty_cache()
        y_ = self.tensor2ndarray(self.model(self.batch[:-1], self.batch[1:]))
        empty_cache()
        self.batch[0] = self.batch[1]
        # print('rter', y_)
        return y_
