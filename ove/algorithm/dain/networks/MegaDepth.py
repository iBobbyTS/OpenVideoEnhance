import torch
import torch.nn as nn
from functools import reduce
from ove.utils.modeling import Sequential


class LambdaBase(Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


def LA(x):
    return x


def LB(x, y, dim=1):
    return torch.cat((x, y), dim)


def LC(x, y):
    return x + y


HourGlass = Sequential(  # Sequential,
    nn.Conv2d(3, 128, (7, 7), (1, 1), (3, 3)),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    Sequential(  # Sequential
        LambdaMap(
            LA,  # ConcatTable
            Sequential(  # Sequential
                nn.MaxPool2d((2, 2), (2, 2)),
                LambdaReduce(
                    LB,  # Concat
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                ),
                LambdaReduce(
                    LB,  # Concat
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                ),
                Sequential(  # Sequential
                    LambdaMap(
                        LA,  # ConcatTable
                        Sequential(  # Sequential
                            nn.MaxPool2d((2, 2), (2, 2)),
                            LambdaReduce(
                                LB,  # Concat
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                            ),
                            LambdaReduce(
                                LB,  # Concat
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 64, (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                            ),
                            Sequential(  # Sequential
                                LambdaMap(
                                    LA,  # ConcatTable
                                    Sequential(  # Sequential
                                        LambdaReduce(
                                            LB,  # Concat
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                        ),
                                        LambdaReduce(
                                            LB,  # Concat
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, (11, 11), (1, 1), (5, 5)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                        ),
                                    ),
                                    Sequential(  # Sequential
                                        nn.AvgPool2d((2, 2), (2, 2)),
                                        LambdaReduce(
                                            LB,  # Concat
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                        ),
                                        LambdaReduce(
                                            LB,  # Concat
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                        ),
                                        Sequential(  # Sequential
                                            LambdaMap(
                                                LA,  # ConcatTable
                                                Sequential(  # Sequential
                                                    LambdaReduce(
                                                        LB,  # Concat
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 64, (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                    ),
                                                    LambdaReduce(
                                                        LB,  # Concat
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 64, (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                    ),
                                                ),
                                                Sequential(  # Sequential
                                                    nn.AvgPool2d((2, 2), (2, 2)),
                                                    LambdaReduce(
                                                        LB,  # Concat
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 64, (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                    ),
                                                    LambdaReduce(
                                                        LB,  # Concat
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 64, (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                    ),
                                                    LambdaReduce(
                                                        LB,  # Concat
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 64, (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                        Sequential(  # Sequential
                                                            nn.Conv2d(256, 32, (1, 1)),
                                                            nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True),
                                                            nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                            nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                            nn.ReLU(inplace=True)
                                                        ),
                                                    ),
                                                    nn.UpsamplingNearest2d(scale_factor=2),
                                                ),
                                            ),
                                            LambdaReduce(LC),  # CAddTable
                                        ),
                                        LambdaReduce(
                                            LB,  # Concat
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential,
                                                nn.Conv2d(256, 32, (1, 1)),
                                                nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            )
                                        ),
                                        LambdaReduce(
                                            LB,  # Concat
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                            Sequential(  # Sequential
                                                nn.Conv2d(256, 64, (1, 1)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(64, 64, (11, 11), (1, 1), (5, 5)),
                                                nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                                nn.ReLU(inplace=True)
                                            ),
                                        ),
                                        nn.UpsamplingNearest2d(scale_factor=2),
                                    ),
                                ),
                                LambdaReduce(LC)  # CAddTable
                            ),
                            LambdaReduce(
                                LB,  # Concat
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 64, (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 64, (5, 5), (1, 1), (2, 2)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 64, (7, 7), (1, 1), (3, 3)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                            ),
                            LambdaReduce(
                                LB,  # Concat
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(256, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                            ),
                            nn.UpsamplingNearest2d(scale_factor=2)
                        ),
                        Sequential(  # Sequential
                            LambdaReduce(
                                LB,  # Concat
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (5, 5), (1, 1), (2, 2)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                            ),
                            LambdaReduce(
                                LB,  # Concat
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 32, (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 64, (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 64, (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 32, (7, 7), (1, 1), (3, 3)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                                Sequential(  # Sequential
                                    nn.Conv2d(128, 64, (1, 1)),
                                    nn.BatchNorm2d(64, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 32, (11, 11), (1, 1), (5, 5)),
                                    nn.BatchNorm2d(32, 1e-05, 0.1, False),
                                    nn.ReLU(inplace=True)
                                ),
                            ),
                        ),
                    ),
                    LambdaReduce(LC),  # CAddTable
                ),
                LambdaReduce(
                    LB,  # Concat
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 64, (1, 1)),
                        nn.BatchNorm2d(64, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 64, (1, 1)),
                        nn.BatchNorm2d(64, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 64, (1, 1)),
                        nn.BatchNorm2d(64, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 32, (7, 7), (1, 1), (3, 3)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                ),
                LambdaReduce(
                    LB,  # Concat
                    Sequential(  # Sequential
                        nn.Conv2d(128, 16, (1, 1)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential,
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 16, (7, 7), (1, 1), (3, 3)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 32, (1, 1)),
                        nn.BatchNorm2d(32, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 16, (11, 11), (1, 1), (5, 5)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                ),
                nn.UpsamplingNearest2d(scale_factor=2)
            ),
            Sequential(  # Sequential
                LambdaReduce(
                    LB,  # Concat
                    Sequential(  # Sequential
                        nn.Conv2d(128, 16, (1, 1)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 64, (1, 1)),
                        nn.BatchNorm2d(64, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 16, (3, 3), (1, 1), (1, 1)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 64, (1, 1)),
                        nn.BatchNorm2d(64, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 16, (7, 7), (1, 1), (3, 3)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                    Sequential(  # Sequential
                        nn.Conv2d(128, 64, (1, 1)),
                        nn.BatchNorm2d(64, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 16, (11, 11), (1, 1), (5, 5)),
                        nn.BatchNorm2d(16, 1e-05, 0.1, False),
                        nn.ReLU(inplace=True)
                    ),
                ),
            ),
        ),
        LambdaReduce(LC)  # CAddTable
    ),
    nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))
)
