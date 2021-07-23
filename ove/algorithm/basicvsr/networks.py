import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from ove.utils.modeling import Sequential
from ove.utils.arch import make_layer


def warp(
        x, flow,
        interpolation='bicubic',
        padding_mode='zeros',
        align_corners=True
):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_flow = torch.stack(torch.meshgrid(
        torch.arange(0, h, requires_grad=False, device=x.device, dtype=torch.int16),
        torch.arange(0, w, requires_grad=False, device=x.device, dtype=torch.int16)
    ), 2).type_as(x)
    grid_flow = grid_flow + flow
    grid_flow *= 2.0
    grid_flow[:, :, :, 0] /= max(w - 1, 1) - 1.0
    grid_flow[:, :, :, 1] /= max(h - 1, 1) - 1.0
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners
    )
    return output


class MyConvModule(ConvModule):
    def forward(self, x, activate=True, norm=True):
        torch.cuda.empty_cache()
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
            torch.cuda.empty_cache()
        return x


class PixelShufflePack(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            scale_factor, upsample_kernel
    ):
        super().__init__()
        self.main = Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * scale_factor * scale_factor,
                upsample_kernel,
                padding=(upsample_kernel - 1) // 2
            ),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv = Sequential(
            nn.Conv2d(mid_channels, mid_channels, (3, 3), (1, 1), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, (3, 3), (1, 1), 1, bias=True)
        )

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x *= self.res_scale
        identity += x
        return identity


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        self.main = Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            *make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        )

    def forward(self, feat):
        return self.main(feat)


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = Sequential(
            MyConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                act_cfg=dict(type='ReLU')
            ),
            MyConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                act_cfg=dict(type='ReLU')
            ),
            MyConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                act_cfg=dict(type='ReLU')
            ),
            MyConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                act_cfg=dict(type='ReLU')
            ),
            MyConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                act_cfg=None)
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.basic_module = Sequential(
            *[SPyNetBasicModule() for _ in range(6)]
        )
        self.load_state_dict(torch.load(pretrained))
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()
        ref -= self.mean
        ref /= self.std
        ref = [ref]
        supp -= self.mean
        supp /= self.std
        supp = [supp]
        for level in range(5):
            ref.append(F.avg_pool2d(
                input=ref[-1],
                kernel_size=2,
                stride=2,
                count_include_pad=False
            ))
            supp.append(F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False
            ))
        ref = ref[::-1]
        supp = supp[::-1]
        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bicubic',
                    align_corners=True
                ) * 2.0
            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref.pop(level),
                    warp(
                        supp.pop(level),
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'
                    ), flow_up
                ], 1))
            torch.cuda.empty_cache()
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref,
            size=(h_up, w_up),
            mode='bicubic',
            align_corners=False
        )
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bicubic',
            align_corners=False
        )
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        del ref, supp
        torch.cuda.empty_cache()
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow
