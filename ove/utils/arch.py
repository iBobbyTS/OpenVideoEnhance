import torch
from torch.nn import init
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from ove_ext.basicsr.dcn import ModulatedDeformConvPack, modulated_deform_conv
except ImportError:
    ModulatedDeformConvPack = object
    modulated_deform_conv = lambda: None


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    return [basic_block(**kwarg) for _ in range(num_basic_block)]


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        )
        if not pytorch_init:
            default_init_weights([self.conv], 0.1)

    def forward(self, x):
        return x + self.conv(x) * self.res_scale


class DCNv2Pack(ModulatedDeformConvPack):
    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deformable_groups
        )
