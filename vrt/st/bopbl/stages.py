# Stage 1
import numpy
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from .Global.options.test_options import TestOptions
from .Global.test import parameter_set, Pix2PixHDModel_Mapping, irregular_hole_synthesize, data_transforms, data_transforms_rgb_old
from .Global.detection_models import networks

# Stage 2
import dlib
from skimage.transform import warp
from .Face_Detection.detect_all_dlib import search

# Stage 3
from .Face_Enhancement.models.pix2pix_model import Pix2PixModel

# Stage 4
import cv2
from .Face_Detection.align_warp_back_multiple_dlib import compute_transformation_matrix, match_histograms, blur_blending_cv2


def cv2_to_pil(frame):
    return Image.fromarray(frame[:, :, ::-1])


class Stage1:
    def __init__(self, with_scratch=False, input_size='full_size'):
        self.opt = TestOptions().parse(save=False)
        parameter_set(self.opt)
        self.model = Pix2PixHDModel_Mapping()
        self.model.initialize(self.opt)
        self.model.eval()
        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.mask_transform = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if with_scratch:
            assert input_size in ('resize_256', 'full_size', 'scale_256')
            self.input_size = input_size
            self.scratch_model = networks.UNet(
                in_channels=1,
                out_channels=1,
                depth=4,
                conv_num=2,
                wf=6,
                padding=True,
                batch_norm=True,
                up_mode='upsample',
                with_tanh=False,
                sync_bn=True,
                antialiasing=True,
            )
            checkpoint_path = "./checkpoints/detection/FT_Epoch_latest.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.scratch_model.load_state_dict(checkpoint["model_state"])
            self.scratch_model.to(self.device)
            self.scratch_model.eval()
            self.process = self.w_scratch
        else:
            self.process = self.wo_scratch

    def inference(self, frame, mask):
        try:
            return (self.model.inference(frame, mask).data.cpu() + 1.0) / 2.0
        except RuntimeError:
            return (frame.data.cpu() + 1.0) / 2.0

    def w_scratch(self, frame: numpy.ndarray):
        frame = cv2_to_pil(frame)
        scratch_image = Image.fromarray(frame)
        transformed_image_PIL = data_transforms(scratch_image, self.input_size)
        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = torchvision.transforms.ToTensor()(scratch_image)
        scratch_image = torchvision.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        scratch_image = scratch_image.to(self.device)
        frame = torch.sigmoid(self.scratch_model(scratch_image))
        frame = frame.data.cpu()
        mask = frame
        frame = irregular_hole_synthesize(frame, mask)
        mask = self.mask_transform(mask)
        mask = mask[:1, :, :]  # Convert to single channel
        mask = mask.unsqueeze(0)
        frame = self.img_transform(frame)
        frame = frame.unsqueeze(0)
        return self.inference(frame, mask)

    def wo_scratch(self, frame):
        frame = cv2_to_pil(frame)
        if self.opt.test_mode == "Scale":
            frame = data_transforms(frame, scale=True)
        if self.opt.test_mode == "Full":
            frame = data_transforms(frame, scale=False)
        if self.opt.test_mode == "Crop":
            frame = data_transforms_rgb_old(frame)
        frame = self.img_transform(frame)
        frame = frame.unsqueeze(0)
        mask = torch.zeros_like(frame)
        return self.inference(frame, mask)


class Stage2:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_locator = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def process(self, frame):
        out = []
        faces = self.face_detector(frame)
        for face in faces:
            face_landmarks = self.landmark_locator(frame, face)
            current_fl = search(face_landmarks)
            affine = compute_transformation_matrix(frame, current_fl, False, target_face_scale=1.3).params
            out.append(warp(frame, affine, output_shape=(256, 256, 3)))
        return out


class Stage3:
    def __init__(self):
        opt = TestOptions().parse()
        self.model = Pix2PixModel(opt)
        self.model.eval()

    def process(self, frame):
        return self.model(frame, mode="inference")


class Stage4:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_locator = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def process(self, frame, stage2_faces):
        faces = self.face_detector(frame)
        blended = frame
        for current_face, stage2_face in zip(faces, stage2_faces):
            face_landmarks = self.landmark_locator(frame, current_face)
            current_fl = search(face_landmarks)
            forward_mask = numpy.ones_like(frame).astype("uint8")
            affine = compute_transformation_matrix(frame, current_fl, False, target_face_scale=1.3)
            aligned_face = warp(frame, affine, output_shape=(256, 256, 3), preserve_range=True)
            forward_mask = warp(forward_mask, affine, output_shape=(256, 256, 3), order=0, preserve_range=True)
            affine_inverse = affine.inverse
            # cur_face = aligned_face
            cur_face = stage2_face
            A = cv2.cvtColor(aligned_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = cv2.cvtColor(cur_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = match_histograms(B, A)
            cur_face = cv2.cvtColor(B.astype("uint8"), cv2.COLOR_BGR2RGB)
            warped_back = warp(
                cur_face,
                affine_inverse,
                output_shape=(self.height, self.width, 3),
                order=3,
                preserve_range=True,
            )

            backward_mask = warp(
                forward_mask,
                affine_inverse,
                output_shape=(self.height, self.width, 3),
                order=0,
                preserve_range=True,
            )  # Nearest neighbour

            blended = blur_blending_cv2(warped_back, blended, backward_mask)
            blended *= 255.0
        return blended
