import os
import numpy as np
import imageio.v2 as imageio

""" Load customized QXSLAB_SAROPT dataset """

_QXSLAB_SAROPT_TEST_IMG_PATH_1 = r'E:\cmm_proj\QXSLAB_SAROPT\test\opt_256_oc_0.2'
_QXSLAB_SAROPT_TEST_IMG_PATH_2 = r'E:\cmm_proj\QXSLAB_SAROPT\test\sar_256_oc_0.2'

def _yield_QXSLAB_SAROPT_img(img_path_1, img_path_2, img_list):
    for img_name in img_list:
        try:
            image_1 = imageio.imread(os.path.join(img_path_1, img_name))
            image_2 = imageio.imread(os.path.join(img_path_2, img_name))
        except:
            image_1 = np.array([])
            image_2 = np.array([])

        yield img_name, image_1, image_2, None


def load_QXSLAB_SAROPT(img_path_1 = _QXSLAB_SAROPT_TEST_IMG_PATH_1, img_path_2 = _QXSLAB_SAROPT_TEST_IMG_PATH_2):
    # Only calculate the file both existed in the two folders
    img_list = list(set(os.listdir(img_path_1)) & set(os.listdir(img_path_2)))
    return len(img_list), lambda : _yield_QXSLAB_SAROPT_img(img_path_1, img_path_2, img_list)

""" Load OSdataset 2.0 dataset """
_OSdataset2_TEST_IMG_PATH = r'E:\cmm_proj\OSDataset2.0\Patch-level Subset\OSdataset\512\val'

def _yield_OSdataset2_img(dataset_path, count):

    for i in range(1, count + 1):
        try:
            image_1 = imageio.imread(os.path.join(dataset_path, f'opt{i}.png'))
            image_2 = imageio.imread(os.path.join(dataset_path, f'sar{i}.png'))
        except:
            image_1 = np.array([])
            image_2 = np.array([])

        yield f'opt{i}.png', image_1, image_2, None

def load_OSdataset2(dataset_path = _OSdataset2_TEST_IMG_PATH):
    image_list = os.listdir(dataset_path)
    return len(image_list), lambda : _yield_OSdataset2_img(dataset_path, len(image_list))
