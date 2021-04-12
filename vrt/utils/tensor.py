import numpy
import torch
import copy

name2module_dict = {'numpy': numpy, 'torch': torch}


class Tensor:
    dtype_precision = {
        'uint8': 0,
        'int8': 0,
        'uint16': 0,
        'int16': 0,
        'float16': 1,
        'float32': 1,
        'float64': 1,
    }
    dtype_speed = {
        'uint8': 5,
        'int8': 5,
        'uint16': 4,
        'int16': 4,
        'float16': 3,
        'float32': 2,
        'float64': 1
    }
    place_speed = {
        'numpy': 0 if torch.cuda.is_available() else 1,
        'torch': 1 if torch.cuda.is_available() else 0
    }
    torch_dtype = {
        'uint8': torch.uint8,
        'int8': torch.int8,
        'float64': torch.float64,
        'float32': torch.float32,
        'float16': torch.float16,
    }

    def __init__(
            self,
            tensor,
            shape_order: str, channel_order: str,
            range_: tuple, clamp=True
    ):
        # Pre store
        self.tensor = tensor
        if isinstance(self.tensor, (list, tuple)):
            self.tensor = name2module(self.tensor[0]).stack(self.tensor)
        # Check if shape orders match tensor shape
        assert len(self.tensor.shape) == len(shape_order), \
            f"Image shape {tuple(tensor.shape)} doesn't match given order ({', '.join(shape_order)})"
        self.place = type(self.tensor).__module__
        # Functions different for numpy and pytorch
        self.shape_order_str2dict = lambda shape_order_: {
            name: index for index, name in enumerate(tuple(shape_order_.lower()))
        }
        # Store variables
        self.channel_order = channel_order.lower()
        """
        Shape order notes: 
        f: frame
        e: empty dim
        b: batch
        c: channel
        h: height
        w: width
        """
        self.shape_order_str = shape_order
        self.shape_order = {name: index for index, name in enumerate(tuple(shape_order.lower()))}
        self.dtype = str(self.tensor.dtype) if self.place == 'numpy' else str(self.tensor.dtype).split('.')[1]
        self.device = None if self.place == 'numpy' else str(self.tensor.device)
        self.min, self.max = range_
        self.shape = self.size()
        if clamp:
            if self.tensor.max() > self.max or self.tensor.min() > self.min:
                self.tensor = self.clamp(self.min, self.max)

    def __len__(self):
        return len(self.tensor)

    def __iter__(self):
        # First dim of tensor will be iterated
        return iter([Tensor(
            tensor=_,
            shape_order=self.shape_order_str[1:],
            channel_order=self.channel_order,
            range_=(self.min, self.max)
        ) for _ in self.tensor])

    def __str__(self):
        string = str(self.tensor)
        if self.place == 'numpy':
            string = "numpy_array(" \
                     f"{string}, " \
                     f"dtype=numpy.{self.dtype}"
        else:
            string = "torch_tensor(" \
                     f"{string[7:-1]}"
        string = string + ", " \
                          f"shape_order={self.shape_order_str}, " \
                          f"channel_order={self.channel_order}, " \
                          f"range=({self.min}, {self.max})" \
                          ")"
        return string

    def __getitem__(self, index):
        return Tensor(
            tensor=self.tensor[index],
            shape_order=self.shape_order_str[1:] if isinstance(index, int) else self.shape_order_str,
            channel_order=self.channel_order,
            range_=(self.min, self.max)
        )

    def __setitem__(self, index, tensor):
        assert tensor.shape == self.tensor[index].shape
        self.tensor[index] = tensor.tensor if isinstance(tensor, Tensor) else tensor

    # Internal methods
    def clamp(self, min_, max_):
        return {
            'numpy': numpy.clip,
            'torch': torch.clamp
        }[self.place](self.tensor, min_, max_)

    # Conversion functions
    def cvt_place(self, place):
        if place is not None:
            if place != self.place:
                if place == 'torch':
                    for i in self.tensor.strides:
                        if i < 0:
                            self.tensor = self.tensor.copy()
                            break
                    self.tensor = torch.from_numpy(self.tensor)
                if place == 'numpy':
                    if str(self.tensor.device) != 'cpu':
                        self.tensor = self.tensor.cpu()
                    self.tensor = self.tensor.numpy()
                self.place = place

    def cvt_device(self, device):
        if device is not None:
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.place == 'torch' and self.device != device:
                self.tensor = self.tensor.to(device)
                self.device = device

    def cvt_shape_order(self, shape_order):
        if shape_order is not None:
            if shape_order != self.shape_order_str:
                if self.place == 'numpy':
                    self.tensor = numpy.transpose(self.tensor, [self.shape_order[_] for _ in shape_order])
                elif self.place == 'torch':
                    self.tensor = self.tensor.permute([self.shape_order[_] for _ in shape_order])
                self.shape_order_str = shape_order
                self.shape_order = self.shape_order_str2dict(self.shape_order_str)

    def cvt_channel_order(self, channel_order):
        if channel_order is not None:
            if channel_order != self.channel_order:
                if channel_order[::-1] == self.channel_order:
                    self.tensor = name2module_dict[self.place].flip(self.tensor, (self.shape_order['c'],))
                    self.channel_order = channel_order
                else:
                    print(f'Unknown conversion: {self.channel_order} to {channel_order}')

    def cvt_dtype(self, dtype):
        if dtype is not None:
            if dtype != self.dtype:
                if self.tensor.max() > self.max or self.tensor.min() > self.min:
                    self.tensor = self.clamp(self.min, self.max)
                if self.place == 'numpy':
                    self.tensor = self.tensor.astype(dtype)
                elif self.place == 'torch':
                    self.tensor = self.tensor.to(self.torch_dtype[dtype])
                self.dtype = dtype

    def cvt_range(self, range_):
        if range_ is not None:
            if ((min_ := range_[0]), (max_ := range_[1])) != (self.min, self.max):
                self.tensor -= self.min
                self.tensor /= self.max - self.min
                self.tensor *= max_ - min_
                self.tensor += min_
                self.min, self.max = min_, max_

    def convert(
            self,
            place=None, device='auto', dtype=None,
            shape_order=None, channel_order=None, range_=None
    ):
        executions = [
            (self.cvt_shape_order, shape_order),
            (self.cvt_channel_order, channel_order)
        ]
        if dtype is not None and self.dtype != dtype:
            if self.dtype_speed[dtype] >= self.dtype_speed[self.dtype]:
                # float32 to uint8, float32 to float16
                if self.dtype_precision[dtype] <= self.dtype_precision[self.dtype]:
                    # *float2int/float2float/int2int
                    executions.insert(0, (self.cvt_range, range_))
                    executions.insert(1, (self.cvt_dtype, dtype))
                else:  # int2float
                    executions.insert(0, (self.cvt_dtype, dtype))
                    executions.insert(1, (self.cvt_range, range_))
            else:  # uint8 to float32, float16 to float32
                if self.dtype_precision[dtype] <= self.dtype_precision[self.dtype]:
                    # *float2int/float2float/int2int
                    executions.extend([
                        (self.cvt_range, range_),
                        (self.cvt_dtype, dtype)
                    ])
                else:  # int2float
                    executions.extend([
                        (self.cvt_dtype, dtype),
                        (self.cvt_range, range_)
                    ])

        if place is not None and self.place != place:
            if self.place_speed[place] == 1:
                executions.insert(0, (self.cvt_place, place))
                executions.insert(1, (self.cvt_device, device))
            else:
                executions.extend([(self.cvt_place, place), (self.cvt_device, device)])
        # Exec
        for func, arg in executions:
            func(arg)
        self.shape = self.size()
        return self.tensor

    def copy(self):
        return copy.deepcopy(self)

    def detach(self):
        if self.place == 'torch':
            self.tensor = self.tensor.detach()

    def size(self):
        return tuple(self.tensor.shape)

    def update(self, tensor):
        self.tensor = tensor
        self.place = type(self.tensor).__module__
        self.dtype = str(self.tensor.dtype) if self.place == 'numpy' else str(self.tensor.dtype).split('.')[1]
        self.device = None if self.place == 'numpy' else str(self.tensor.device)
        self.shape = self.size()

    def unsqueeze(self, index=0, name='b'):
        if self.place == 'numpy':
            self.tensor = numpy.expand_dims(self.tensor, index)
        else:
            self.tensor = self.tensor.unsqueeze(index)
        self.shape_order_str = f"{self.shape_order_str[:index]}{name}{self.shape_order_str[index:]}"
        self.shape_order = self.shape_order_str2dict(self.shape_order_str)
        self.shape = self.size()



def stack(tensors: list):
    assert len(set([tuple(map(
        _.__getattribute__, ('shape_order_str', 'channel_order', 'min', 'max', 'place', 'device', 'dtype'))
    ) for _ in tensors])) == 1, \
        'Tensors has different properties'
    return Tensor(
        tensor=name2module_dict[tensors[0].place].stack([_.tensor for _ in tensors]),
        shape_order='f' + tensors[0].shape_order_str,
        channel_order=tensors[0].channel_order,
        range_=(tensors[0].min, tensors[0].max)
    )


def name2module(obj):
    return name2module_dict[type(obj).__module__]
