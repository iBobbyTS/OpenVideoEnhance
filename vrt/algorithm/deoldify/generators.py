import os
from typing import Optional, Tuple, Any, Callable

from fastai import vision
from fastai import layers as fastai_layers
from fastai import core, torch_core, basic_data, basic_train

from . import dataset
from . import unet


# Wide
def unet_learner_wide(
        data: basic_data.DataBunch,
        arch: Callable,
        pretrained: bool = True,
        blur_final: bool = True,
        norm_type: Optional[fastai_layers.NormType] = fastai_layers.NormType,
        split_on: Optional[torch_core.SplitFuncOrIdxList] = None,
        blur: bool = False,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        nf_factor: int = 1,
        **kwargs: Any
) -> basic_train.Learner:
    meta = vision.learner.cnn_config(arch)
    body = vision.create_body(arch, pretrained)
    model = torch_core.to_device(
        unet.DynamicUnetWide(
            body,
            n_classes=data.c,
            blur=blur,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = basic_train.Learner(data, model, **kwargs)
    learn.split(core.ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    torch_core.apply_init(model[2], torch_core.nn.init.kaiming_normal_)
    return learn


def gen_learner_wide(
        data: vision.data.ImageDataBunch, gen_loss, arch=vision.models.resnet101, nf_factor: int = 2
) -> basic_train.Learner:
    return unet_learner_wide(
        data,
        arch=arch,
        wd=1e-3,
        blur=True,
        norm_type=fastai_layers.NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


def gen_inference_wide(
        temp_path, state_dict, nf_factor: int = 2, arch=vision.models.resnet101
) -> basic_train.Learner:
    data = dataset.get_dummy_databunch(temp_path)
    learn = gen_learner_wide(
        data=data, gen_loss=torch_core.F.l1_loss, nf_factor=nf_factor, arch=arch
    )
    learn.model.load_state_dict(state_dict)
    learn.model.eval()
    return learn


# Deep
def unet_learner_deep(
    data: basic_data.DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[fastai_layers.NormType] = fastai_layers.NormType,
    split_on: Optional[torch_core.SplitFuncOrIdxList] = None,
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    nf_factor: float = 1.5,
    **kwargs: Any
) -> basic_train.Learner:
    meta = vision.learner.cnn_config(arch)
    body = vision.create_body(arch, pretrained)
    model = torch_core.to_device(
        unet.DynamicUnetDeep(
            body,
            n_classes=data.c,
            blur=blur,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
            nf_factor=nf_factor,
        ),
        data.device,
    )
    learn = basic_train.Learner(data, model, **kwargs)
    learn.split(core.ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    torch_core.apply_init(model[2], torch_core.nn.init.kaiming_normal_)
    return learn


def gen_learner_deep(
        data: vision.data.ImageDataBunch, gen_loss, arch=vision.models.resnet34, nf_factor: float = 1.5
) -> basic_train.Learner:
    return unet_learner_deep(
        data,
        arch,
        wd=1e-3,
        blur=True,
        norm_type=fastai_layers.NormType.Spectral,
        self_attention=True,
        y_range=(-3.0, 3.0),
        loss_func=gen_loss,
        nf_factor=nf_factor,
    )


def gen_inference_deep(
        state_dict, arch=vision.models.resnet34, nf_factor: float = 1.5
) -> basic_train.Learner:
    data = dataset.get_dummy_databunch()
    learn = gen_learner_deep(
        data=data, gen_loss=torch_core.F.l1_loss, arch=arch, nf_factor=nf_factor
    )
    learn.model.load_state_dict(state_dict)
    learn.model.eval()
    return learn
