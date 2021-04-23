import functools

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "spectral":
        norm_layer = spectral_norm()
    # elif norm_type == "SwitchNorm":
    #     norm_layer = SwitchNorm2d
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResnetBlock(nn.Module):
    def __init__(
            self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, dilation=1
    ):
        super().__init__()
        self.dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == "zero":
            p = self.dilation
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=1), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class GlobalGenerator_DCDCv2(nn.Module):
    def __init__(
            self,
            input_nc,
            output_nc,
            ngf=64,
            k_size=3,
            n_downsampling=8,
            norm_layer=nn.BatchNorm2d,
            padding_type="reflect"
    ):
        super().__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, min(ngf, 64), kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        # Down sample
        for i in range(1):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, 64),
                    min(ngf * mult * 2, 64),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, 64)),
                activation,
            ]
        for i in range(1, n_downsampling - 1):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, 64),
                    min(ngf * mult * 2, 64),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, 64)),
                activation,
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, 64),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, 64),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer
                )
            ]
        mult = 2 ** (n_downsampling - 1)

        model += [
            ResnetBlock(
                min(ngf * mult * 2, 64),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer
            )
        ]
        model += [
            ResnetBlock(
                min(ngf * mult * 2, 64),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer
            )
        ]
        self.encoder = nn.Sequential(*model)

        # decode
        model = []
        o_pad = 0 if k_size == 4 else 1
        mult = 2 ** n_downsampling
        model += [
            ResnetBlock(
                min(ngf * mult, 64),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer
            )
        ]

        model += [
            ResnetBlock(
                min(ngf * mult, 64),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer
            )
        ]

        for i in range(1, n_downsampling - 1):
            mult = 2 ** (n_downsampling - i)
            model += [
                ResnetBlock(
                    min(ngf * mult, 64),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult, 64),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer
                )
            ]
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, 64),
                    min(int(ngf * mult / 2), 64),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), 64)),
                activation,
            ]
        for i in range(n_downsampling - 1, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, 64),
                    min(int(ngf * mult / 2), 64),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), 64)),
                activation,
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(min(ngf, 64), output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*model)

    def forward(self, input, flow="enc_dec"):
        if flow == "enc":
            return self.encoder(input)
        elif flow == "dec":
            return self.decoder(input)
        elif flow == "enc_dec":
            x = self.encoder(input)
            x = self.decoder(x)
            return x


class NonLocalBlock2D_with_mask_Res(nn.Module):
    def __init__(
            self,
            in_channels,
            inter_channels,
            mode="add",
            re_norm=False,
            temperature=1.0,
            use_self=False,
            cosin=False,
    ):
        super().__init__()

        self.cosin = cosin
        self.renorm = re_norm
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.W = nn.Conv2d(
            in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0
        )
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.mode = mode
        self.temperature = temperature
        self.use_self = use_self

        norm_layer = get_norm_layer(norm_type="instance")
        activation = nn.ReLU(True)

        model = []
        for i in range(3):
            model += [
                ResnetBlock(
                    inter_channels,
                    padding_type="reflect",
                    activation=activation,
                    norm_layer=norm_layer
                )
            ]
        self.res_block = nn.Sequential(*model)

    def forward(self, x, mask):  ## The shape of mask is Batch*1*H*W
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        if self.cosin:
            theta_x = F.normalize(theta_x, dim=2)
            phi_x = F.normalize(phi_x, dim=1)

        f = torch.matmul(theta_x, phi_x)

        f /= self.temperature

        f_div_C = F.softmax(f, dim=2)

        tmp = 1 - mask
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        mask[mask > 0] = 1.0
        mask = 1 - mask

        tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        mask *= tmp

        mask_expand = mask.view(batch_size, 1, -1)
        mask_expand = mask_expand.repeat(1, x.size(2) * x.size(3), 1)

        if self.use_self:
            mask_expand[:, range(x.size(2) * x.size(3)), range(x.size(2) * x.size(3))] = 1.0

        f_div_C = mask_expand * f_div_C
        if self.renorm:
            f_div_C = F.normalize(f_div_C, p=1, dim=2)

        ###########################

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        W_y = self.res_block(W_y)

        if self.mode == "combine":
            full_mask = mask.repeat(1, self.inter_channels, 1, 1)
            z = full_mask * x + (1 - full_mask) * W_y
            return z


class Mapping_Model_with_mask(nn.Module):
    def __init__(self, mc=64, n_blocks=3, norm="instance", padding_type="reflect"):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []

        for i in range(4):
            ic = min(64 * (2 ** i), mc)
            oc = min(64 * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]

        self.before_NL = nn.Sequential(*model)

        self.NL = NonLocalBlock2D_with_mask_Res(
            mc,
            mc,
            mode='combine',
            re_norm=True,
            temperature=1.0,
            use_self=False,
            cosin=False,
        )

        model = []
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    mc,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    dilation=1
                )
            ]

        for i in range(3):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(128, 64, 3, 1, 1)]
        self.after_NL = nn.Sequential(*model)

    def forward(self, input, mask):
        x = self.before_NL(input)
        del input
        x = self.NL(x, mask)
        del mask
        x = self.after_NL(x)
        return x


class Mapping_Model(nn.Module):
    def __init__(self, mc=64, n_blocks=3, norm="instance", padding_type="reflect"):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []

        for i in range(4):
            ic = min(64 * (2 ** i), mc)
            oc = min(64 * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    mc,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    dilation=1,
                )
            ]

        for i in range(63):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(128, 64, 3, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Pix2PixHDModel_Mapping(nn.Module):
    name = "Pix2PixHDModel_Mapping"

    def __init__(
            self,
            with_scratch,
            state_dict
    ):
        super().__init__()
        self.with_scratch = with_scratch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cuda_availability = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.cuda_availability else torch.Tensor
        # define models
        self.netG_A = GlobalGenerator_DCDCv2(
            input_nc=3,
            output_nc=3,
            k_size=4,
            n_downsampling=3,
            norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)
        )
        self.netG_B = GlobalGenerator_DCDCv2(
            input_nc=3,
            output_nc=3,
            k_size=4,
            n_downsampling=3,
            norm_layer=functools.partial(nn.InstanceNorm2d, affine=False)
        )
        self.mapping_net = {
            True: Mapping_Model_with_mask,
            False: Mapping_Model
        }[bool(with_scratch)](mc=512, n_blocks=6)
        self.mapping_net.apply(weights_init)
        self.netG_A.load_state_dict(state_dict['VAE_A'])
        self.netG_B.load_state_dict(state_dict['VAE_B'])
        for param in self.netG_A.parameters():
            param.requires_grad = False
        for param in self.netG_B.parameters():
            param.requires_grad = False
        self.netG_A.eval()
        self.netG_B.eval()
        if torch.cuda.is_available():
            self.netG_A.cuda()
            self.netG_B.cuda()
            self.mapping_net.cuda()
        self.mapping_net.load_state_dict(state_dict['mapping_net'])

    def inference(self, label, inst):
        input_concat = label.data.to(self.device)
        inst_data = inst.to(self.device)
        label_feat = self.netG_A.forward(input_concat, flow="enc")
        if self.with_scratch:
            label_feat_map = self.mapping_net(label_feat.detach(), inst_data)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())
        fake_image = self.netG_B.forward(label_feat_map, flow="dec")
        return fake_image
