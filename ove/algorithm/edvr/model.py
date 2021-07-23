import torch
import torch.nn as nn
from torch.nn import functional as F
from ove.utils.arch import make_layer

from .networks import PredeblurModule, ResidualBlockNoBN, PCDAlignment, TSAFusion
from ove.utils.modeling import Sequential


class EDVR(nn.Module):
    def __init__(
            self,
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_frame=5,
            deformable_groups=8,
            num_extract_block=5,
            num_reconstruct_block=10,
            center_frame_idx=None,
            hr_in=False,
            with_predeblur=False,
            with_tsa=True,
            scale_factor=4
    ):
        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa
        self.scale_factor = scale_factor

        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
            self.conv_1x1 = nn.Conv2d(num_feat, num_feat, (1, 1), (1, 1))
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, (3, 3), (1, 1), 1)

        # extrat pyramid features
        self.feature_extraction = Sequential(*make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat))
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, (3, 3), (2, 2), 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, (3, 3), (1, 1), 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, (3, 3), (2, 2), 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, (3, 3), (1, 1), 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, (1, 1), (1, 1))

        # reconstruction
        self.reconstruction = Sequential(*make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat))
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, (3, 3), (1, 1), 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, (3, 3), (1, 1), 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, (3, 3), (1, 1), 1)
        self.conv_last = nn.Conv2d(64, 3, (3, 3), (1, 1), 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        b, t, c, h, w = x.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, 'The height and width must be multiple of 16.'
        else:
            assert h % 4 == 0 and w % 4 == 0, 'The height and width must be multiple of 4.'

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out += base
        return out
