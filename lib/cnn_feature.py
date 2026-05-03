import cv2
import numpy as np
import torch

from lib.model import D2Net
from lib.pyramid import process_multiscale
from lib.utils import preprocess_image

_MODEL_FILE = "models/d2_tf.pth"
_CBAM_WIDGET_FILE = "models/cbam_tmrl_final.pth"
_DEFAULT_SCALES = [0.25, 0.50, 1.0]
_MAX_EDGE = 2500
_MAX_SUM_EDGES = 5000


def _resolve_device(device=None):
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def create_feature_extractor(device=None, model_file=_MODEL_FILE, cbam_weight_file=_CBAM_WIDGET_FILE):
    target_device = _resolve_device(device)
    extractor = D2Net(
        model_file=model_file,
        cbam_weight_file=cbam_weight_file,
        use_relu=True,
        use_cuda=(target_device.type == "cuda")
    )
    extractor.eval()
    if target_device.type == "cuda":
        extractor = extractor.to(target_device)
    return extractor, target_device


_DEFAULT_MODEL, _DEFAULT_DEVICE = create_feature_extractor()


def _resize_image(image, scale):
    resized_height = max(1, int(image.shape[0] * scale))
    resized_width = max(1, int(image.shape[1] * scale))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype("float")


def _extract_with_model(image, model, device, scales, nfeatures):
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    resized_image = image
    if max(resized_image.shape) > _MAX_EDGE:
        resized_image = _resize_image(
            resized_image, _MAX_EDGE / max(resized_image.shape)
        )
    if sum(resized_image.shape[:2]) > _MAX_SUM_EDGES:
        resized_image = _resize_image(
            resized_image, _MAX_SUM_EDGES / sum(resized_image.shape[:2])
        )

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image, preprocessing="torch")
    with torch.no_grad():
        keypoints, scores, descriptors = process_multiscale(
            torch.tensor(
                input_image[np.newaxis, :, :, :].astype(np.float32),
                device=device,
            ),
            model,
            scales,
        )

    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    keypoints = keypoints[:, [1, 0, 2]]

    if nfeatures != -1:
        scores2 = np.array([scores]).T
        res = np.hstack((scores2, keypoints))
        res = res[np.lexsort(-res[:, ::-1].T)]

        res = np.hstack((res, descriptors))
        scores = res[0:nfeatures, 0].copy()
        keypoints = res[0:nfeatures, 1:4].copy()
        descriptors = res[0:nfeatures, 4:].copy()
        del res

    return keypoints, scores, descriptors


def cnn_feature_extract_with_extractor(
    image,
    model,
    device,
    scales=None,
    nfeatures=1000,
):
    if scales is None:
        scales = _DEFAULT_SCALES
    return _extract_with_model(image, model, device, scales, nfeatures)


def cnn_feature_extract(image, scales=None, nfeatures=1000):
    if scales is None:
        scales = _DEFAULT_SCALES
    return _extract_with_model(
        image, _DEFAULT_MODEL, _DEFAULT_DEVICE, scales, nfeatures
    )
