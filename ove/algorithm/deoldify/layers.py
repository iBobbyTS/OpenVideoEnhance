from typing import Optional, Callable
from fastai import torch_core
from fastai import layers as fastai_layers


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: Optional[fastai_layers.NormType] = fastai_layers.NormType.Batch,
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = torch_core.nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
    **kwargs
):
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (fastai_layers.NormType.Batch, fastai_layers.NormType.BatchZero) or extra_bn == True
    if bias is None:
        bias = not bn
    conv_func = torch_core.nn.ConvTranspose2d if transpose else torch_core.nn.Conv1d if is_1d else torch_core.nn.Conv2d
    conv = torch_core.init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == fastai_layers.NormType.Weight:
        conv = torch_core.weight_norm(conv)
    elif norm_type == fastai_layers.NormType.Spectral:
        conv = torch_core.spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(fastai_layers.relu(True, leaky=leaky))
    if bn:
        layers.append((torch_core.nn.BatchNorm1d if is_1d else torch_core.nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(fastai_layers.SelfAttention(nf))
    return torch_core.nn.Sequential(*layers)
