import torch
from torch import nn
import torch.nn.functional as F

from ove_ext.basicsr.dcn import ModulatedDeformConvPack, modulated_deform_conv
from ove.utils.arch import default_init_weights, make_layer
from ove.utils.modeling import Sequential


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv = Sequential(
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


class PCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super().__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups
            )
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(
                    num_feat * 2, num_feat, 3, 1, 1
                )

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups
        )

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.relu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.relu(self.offset_conv2[level](offset))
            else:
                offset = self.relu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.relu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.relu(feat)

            if i > 1:
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.relu(
            self.cas_offset_conv2(self.relu(self.cas_offset_conv1(offset))))
        feat = self.relu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super().__init__()
        self.center_frame_idx = center_frame_idx
        # Common
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = Sequential(
            nn.Conv2d(num_frame * num_feat, num_feat, 1, 1),
            self.relu
        )

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = Sequential(
            nn.Conv2d(num_frame * num_feat, num_feat, 1),
            self.relu
        )
        self.spatial_attn2 = Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 1),
            self.relu
        )
        self.spatial_attn3 = Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            self.relu
        )
        self.spatial_attn4 = Sequential(
            nn.Conv2d(num_feat, num_feat, 1),
            self.relu,
            self.upsample,
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )
        self.spatial_attn_l1 = Sequential(
            nn.Conv2d(num_feat, num_feat, 1),
            self.relu
        )
        self.spatial_attn_l2 = Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
            self.relu,
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            self.relu,
            self.upsample
        )
        self.spatial_attn_add = Sequential(
            nn.Conv2d(num_feat, num_feat, 1),
            self.relu,
            nn.Conv2d(num_feat, num_feat, 1)
        )

    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.shape
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone()
        )
        embedding = self.temporal_attn2(
            aligned_feat.view(-1, c, h, w)
        ).view(b, t, -1, h, w)
        # Correlation
        corr_l = [torch.sum(embedding[:, i, :, :, :] * embedding_ref, 1).unsqueeze(1) for i in range(t)]
        corr_prob = torch.sigmoid(
            torch.cat(corr_l, dim=1)
        ).unsqueeze(2).expand(b, t, c, h, w).contiguous().view(b, -1, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
        # Fusion
        feat = self.feat_fusion(aligned_feat)
        # Spatial attention
        attn = self.spatial_attn1(aligned_feat)
        attn = self.spatial_attn2(torch.cat(
            [self.max_pool(attn), self.avg_pool(attn)],
            dim=1
        ))
        # pyramid levels
        attn_level = self.spatial_attn_l1(attn)
        attn_level = self.spatial_attn_l2(torch.cat(
            [self.max_pool(attn_level), self.avg_pool(attn_level)],
            dim=1
        ))

        attn = self.spatial_attn3(attn) + attn_level
        attn = self.spatial_attn4(attn)
        attn_add = self.spatial_attn_add(attn)
        attn = torch.sigmoid(attn)

        feat = feat * attn * 2 + attn_add
        return feat


class PredeblurModule(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super().__init__()
        # Common
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_first = Sequential(
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            self.relu,
            *([] if hr_in else [
                nn.Conv2d(num_feat, num_feat, 3, 2, 1),
                self.relu,
                nn.Conv2d(num_feat, num_feat, 3, 2, 1),
                self.relu
            ])
        )

        # Generate feature pyramid
        self.stride_conv_l2 = Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 2, 1),
            self.relu
        )
        self.stride_conv_l3 = Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 2, 1),
            self.relu,
            ResidualBlockNoBN(num_feat=num_feat),
            self.upsample
        )

        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = Sequential(
            ResidualBlockNoBN(num_feat=num_feat),
            self.upsample
        )
        self.resblock_l1_1 = Sequential(
            *[ResidualBlockNoBN(num_feat=num_feat) for i in range(2)]
        )
        self.resblock_l1_2 = Sequential(
            *[ResidualBlockNoBN(num_feat=num_feat) for i in range(3)]
        )

    def forward(self, x):
        feat_l1 = self.conv_first(x)
        feat_l2 = self.stride_conv_l2(feat_l1)
        feat_l3 = self.stride_conv_l3(feat_l2)
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.resblock_l2_2(feat_l2)

        feat_l1 = self.resblock_l1_1(feat_l1)
        feat_l1 = feat_l1 + feat_l2
        feat_l1 = self.resblock_l1_2(feat_l1)
        return feat_l1


class EDVR(nn.Module):
    def __init__(
        self,
        num_feat=64,
        num_frame=5,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=10,
        center_frame_idx=None,
        hr_in=False,
        with_predeblur=False,
        with_tsa=True
    ):
        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa
        # activation function
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_l1 = Sequential(
            *([
                PredeblurModule(num_feat=num_feat, hr_in=self.hr_in),
                nn.Conv2d(num_feat, num_feat, 1, 1)
            ] if with_predeblur else [
                nn.Conv2d(3, num_feat, 3, 1, 1),
                self.relu
            ]),
            *make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        )
        # Extract pyramid features
        self.conv_l2 = Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 2, 1),
            self.relu,
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            self.relu
        )
        self.conv_l3 = Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 2, 1),
            self.relu,
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            self.relu
        )

        # pcd and tsa module
        self.pcd_align = PCDAlignment(
            num_feat=num_feat, deformable_groups=deformable_groups)
        self.conv_last = Sequential(
            TSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx
            ) if self.with_tsa else nn.Conv2d(
                num_frame * num_feat, num_feat, 1, 1
            ),
            make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat),
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            self.pixel_shuffle,
            self.relu,
            nn.Conv2d(num_feat, 64 * 4, 3, 1, 1),
            self.pixel_shuffle,
            self.relu,
            nn.Conv2d(64, 64, 3, 1, 1),
            self.relu,
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, 'The height and width must be multiple of 16.'
        else:
            assert h % 4 == 0 and w % 4 == 0, 'The height and width must be multiple of 4.'

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.conv_l1(x.view(-1, c, h, w))
        if self.with_predeblur and self.hr_in:
            h //= 4
            w //= 4
        # L2
        feat_l2 = self.conv_l2(feat_l1)
        # L3
        feat_l3 = self.conv_l3(feat_l2)

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = torch.stack([
            self.pcd_align([
                feat_l1[:, i, :, :, :].clone(),
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ], ref_feat_l) for i in range(t)
        ], dim=1)
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        out = self.conv_last(aligned_feat)
        out += x_center if self.hr_in else F.interpolate(
            x_center, scale_factor=4, mode='bilinear', align_corners=False
        )
        return out
