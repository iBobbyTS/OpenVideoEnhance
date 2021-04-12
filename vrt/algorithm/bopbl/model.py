import torch
from torch.nn import functional as F

from vrt import utils
from . import network


# Stage 1
class ScratchDetect:
    def __init__(
            self,
            state_dict,
            height, width
    ):
        # Store args
        self.ori_w, self.ori_h = width, height
        # Initialize pader
        self.pader = utils.modeling.Pader(
            self.ori_w, self.ori_h, 16
        )
        self.padded_w, self.padded_h = self.pader.padded_size
        # Initialize model
        self.model = network.scratch_detect.UNet(
            in_channels=1,
            out_channels=1,
            depth=4
        )
        self.cuda_availability = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_availability else 'cpu')
        # load model
        self.model.load_state_dict(state_dict["model_state"])
        if self.cuda_availability:
            self.model.to(self.device)
        self.model.eval()

    def dt(self, frame):
        frame = torch.mean(frame, dim=1).unsqueeze(0)
        with torch.no_grad():
            frame = torch.sigmoid(self.model(frame))
        frame = F.interpolate(
            frame.data,
            (self.padded_h, self.padded_w),
            mode='bicubic', align_corners=True
        )
        torch.cuda.empty_cache()
        return frame


# Stage 1
class QualityRestore:
    def __init__(self, with_scratch, state_dict, height, width):
        self.with_scratch = with_scratch
        self.model = network.quality_restore.Pix2PixHDModel_Mapping(self.with_scratch, state_dict)
        # Pader
        self.pader = utils.modeling.Pader(
            width, height, 4, extend_func='replication'
        )

    def rt(self, frame, mask=None):
        if self.with_scratch:
            mask = (mask >= 0.25).float()
            frame = frame * (1 - mask) + mask
        else:
            frame = self.pader.pad(frame)
            mask = torch.zeros_like(frame)
        return self.model.inference(frame, mask)


# Stage 2
class FaceDetect:
    def __init__(self, model_path):
        self.model = network.face_detect.FaceDetect(model_path)

    def dt(self, frame):
        faces = self.model(frame)
        faces = torch.stack([torch.from_numpy(face) for face in faces])
        faces -= 0.5
        faces *= 2
        faces = faces.float()
        faces = faces.permute(0, 3, 1, 2)
        return faces


# Stage 3
class FaceRestore:
    def __init__(self, state_dict):
        self.model = network.face_restore.Pix2PixModel(state_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def rt(self, faces):
        return [self.model(face) for face in faces]


# Stage 4
class WarpBack:
    def __init__(self, model_path, height, width):
        self.model = network.face_detect.WrapBack(model_path, height, width)

    def rt(self, frame, faces):
        return self.model(frame, faces)


# Combine
class OverallRestore:
    def __init__(
            self,
            model_path, default_model_dir, device,
            with_scratch,
            height, width
    ):
        # Load state dict
        model_path = {k: utils.folder.check_model(default_model_dir, model_path, v) for k, v in utils.dictionaries.model_paths['bopbl'].items()}
        # model_path = utils.folder.check_model(default_model_dir, model_path, )
        state_dict = torch.load(model_path['common'], map_location=device)
        second_state_dict = torch.load(
            model_path['with_scratch' if with_scratch else 'no_scratch'], map_location=device
        )
        for k in ('QualityRestore', 'ScratchDetect'):
            state_dict[k].update(second_state_dict[k])
        # Scratch Detect
        self.with_scratch = with_scratch
        if self.with_scratch:
            self.scratch_detect_model = ScratchDetect(
                state_dict['ScratchDetect'], height, width
            )
        # Quality Restore
        self.quality_restore_model = QualityRestore(
            with_scratch, state_dict['QualityRestore'],
            height, width
        )
        # FaceDetect
        self.face_detect = FaceDetect(model_path['face_detect'])
        # FaceRestore
        self.face_restore = FaceRestore(state_dict['FaceRestore'])
        # WarpBack
        self.warp_back = WarpBack(model_path['face_detect'], height, width)

    def rt(self, frame):
        # ScratchDetect
        mask = self.scratch_detect_model.dt(frame)[:, 0:1] if self.with_scratch else None
        # QualityRestore
        frame = self.quality_restore_model.rt(frame, mask)
        # FaceDetect
        frame = ((frame+1)*127.5).squeeze(0).permute(1, 2, 0).byte().numpy()
        faces = self.face_detect.dt(frame)
        faces = self.face_restore.rt(faces)
        frame = self.warp_back.rt(frame, faces)
        return frame
