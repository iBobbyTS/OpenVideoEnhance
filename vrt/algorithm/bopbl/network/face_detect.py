import numpy
from skimage.transform import SimilarityTransform, warp
import dlib
import cv2

_standard_face_pts = numpy.array([
    [-0.234375, -0.1171875],
    [0.234375, -0.1171875],
    [0, 0.1171875],
    [-0.140625, 0.4078124761581420898],
    [0.140625, 0.4078124761581420898]
], dtype=numpy.float32)


def get_landmark(face_landmarks, id):
    part = face_landmarks.part(id)
    return part.x, part.y


def search(face_landmarks):
    x1, y1 = get_landmark(face_landmarks, 36)
    x2, y2 = get_landmark(face_landmarks, 39)
    x3, y3 = get_landmark(face_landmarks, 42)
    x4, y4 = get_landmark(face_landmarks, 45)

    x_nose, y_nose = get_landmark(face_landmarks, 30)

    x_left_mouth, y_left_mouth = get_landmark(face_landmarks, 48)
    x_right_mouth, y_right_mouth = get_landmark(face_landmarks, 54)

    x_left_eye = int((x1 + x2) / 2)
    y_left_eye = int((y1 + y2) / 2)
    x_right_eye = int((x3 + x4) / 2)
    y_right_eye = int((y3 + y4) / 2)

    results = numpy.array(
        [
            [x_left_eye, y_left_eye],
            [x_right_eye, y_right_eye],
            [x_nose, y_nose],
            [x_left_mouth, y_left_mouth],
            [x_right_mouth, y_right_mouth],
        ]
    )

    return results


def compute_transformation_matrix(img, landmark, normalize, return_params, target_face_scale=1.0):
    std_pts = _standard_face_pts
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    h, w, c = img.shape
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    affine = SimilarityTransform()

    affine.estimate(target_pts, landmark)
    if return_params:
        return affine.params
    else:
        return affine


class FaceDetect:
    def __init__(self, model_path):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_locator = dlib.shape_predictor(model_path)

    def detect_faces(self, frame):
        faces = self.face_detector(frame)
        if faces:
            return faces
        else:
            return

    def __call__(self, frame):
        returning_faces = []
        for face in self.detect_faces(frame):
            face_landmarks = self.landmark_locator(frame, face)
            current_fl = search(face_landmarks)
            affine = compute_transformation_matrix(frame, current_fl, False, return_params=True, target_face_scale=1.3)
            aligned_face = warp(frame, affine, output_shape=(256, 256, 3))
            returning_faces.append(aligned_face)
        return returning_faces


def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = numpy.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    histogram = lambda ch: numpy.histogram(ch.flatten(), 256, [0, 256])[0]
    src_hist_blue = histogram(src_b)
    src_hist_green = histogram(src_g)
    src_hist_red = histogram(src_r)
    ref_hist_blue = histogram(ref_b)
    ref_hist_green = histogram(ref_g)
    ref_hist_red = histogram(ref_r)
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def blur_blending_cv2(im1, im2, mask):
    mask *= 255.0

    kernel = numpy.ones((9, 9), numpy.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)

    mask_blur = cv2.GaussianBlur(mask, (25, 25), 0)
    mask_blur /= 255.0

    im = im1 * mask_blur + (1 - mask_blur) * im2

    im /= 255.0
    im = numpy.clip(im, 0.0, 1.0)

    return im


class WrapBack(FaceDetect):
    def __init__(self, model_path, height, width):
        super().__init__(model_path)
        self.ori_height = height
        self.ori_width = width

    def __call__(self, *args):
        frame, hr_faces = args
        hr_faces = numpy.array([hr_face.numpy() for hr_face in hr_faces])
        hr_faces = hr_faces.squeeze(1)
        hr_faces += 1
        hr_faces *= 127.5
        hr_faces = hr_faces.astype(numpy.uint8)
        hr_faces = numpy.transpose(hr_faces, (0, 2, 3, 1))
        blended = frame
        for face, hr_face in zip(self.detect_faces(frame), hr_faces):
            face_landmarks = self.landmark_locator(frame, face)
            current_fl = search(face_landmarks)
            affine = compute_transformation_matrix(frame, current_fl, False, return_params=False, target_face_scale=1.3)
            aligned_face = warp(frame, affine, output_shape=(256, 256, 3), preserve_range=True)
            forward_mask = warp(
                numpy.ones_like(frame).astype("uint8"),
                affine, output_shape=(256, 256, 3), order=0, preserve_range=True
            )

            affine_inverse = affine.inverse

            # Histogram Color matching
            A = cv2.cvtColor(aligned_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = cv2.cvtColor(hr_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            B = match_histograms(B, A)
            cur_face = cv2.cvtColor(B.astype("uint8"), cv2.COLOR_BGR2RGB)

            warped_back = warp(
                cur_face,
                affine_inverse,
                output_shape=(self.ori_height, self.ori_width, 3),
                order=3,
                preserve_range=True,
            )

            backward_mask = warp(
                forward_mask,
                affine_inverse,
                output_shape=(self.ori_height, self.ori_width, 3),
                order=0,
                preserve_range=True,
            )  # Nearest neighbour

            blended = blur_blending_cv2(warped_back, blended, backward_mask)
            blended *= 255.0
        return blended
