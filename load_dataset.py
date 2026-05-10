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

        yield img_name.rstrip('.png'), image_1, image_2, None


def load_QXSLAB_SAROPT(img_path_1 = _QXSLAB_SAROPT_TEST_IMG_PATH_1, img_path_2 = _QXSLAB_SAROPT_TEST_IMG_PATH_2):
    # Only calculate the file both existed in the two folders
    img_list = list(set(os.listdir(img_path_1)) & set(os.listdir(img_path_2)))
    return len(img_list), lambda : _yield_QXSLAB_SAROPT_img(img_path_1, img_path_2, img_list)


""" Load OSdataset 2.0 dataset """

_OSdataset2_TEST_IMG_PATH = r'E:\cmm_proj\OSDataset2.0\Patch-level Subset\OSdataset\512\test'

def _yield_OSdataset2_img(dataset_path, count):
    for i in range(1, count + 1):
        try:
            image_1 = imageio.imread(os.path.join(dataset_path, f'opt{i}.png'))
            image_2 = imageio.imread(os.path.join(dataset_path, f'sar{i}.png'))
        except:
            image_1 = np.array([])
            image_2 = np.array([])

        yield f'image_pair_{i}', image_1, image_2, None

def load_OSdataset2(dataset_path = _OSdataset2_TEST_IMG_PATH):
    image_list = os.listdir(dataset_path)
    image_count = int(len(image_list) / 2)
    return image_count, lambda : _yield_OSdataset2_img(dataset_path, image_count)


""" Load df-ms-data dataset """

_DFMsData_PATH = r'.\df-ms-data'

def _yield_DFMsData_img(dataset_path, file_dict):
    print('[Warning] The affine transformation matrix is missing. The evaluation result will be unreliable.')
    for subdir in file_dict.keys():
        full_subdir = os.path.join(dataset_path, subdir)
        for i in range(len(file_dict[subdir]) - 1):
            for k in range(i + 1, len(file_dict[subdir])):
                try:
                    image_1 = imageio.imread(os.path.join(full_subdir, f'{file_dict[subdir][i]}'))
                    image_2 = imageio.imread(os.path.join(full_subdir, f'{file_dict[subdir][k]}'))
                except:
                    image_1 = np.array([])
                    image_2 = np.array([])

                yield f'{subdir}_{i}_{k}', image_1, image_2, None

def load_DFMsData(dataset_path = _DFMsData_PATH):
    img_dict = {}
    pair_count = 0

    for subdir in os.listdir(dataset_path):
        full_subdir = os.path.join(dataset_path, subdir)
        img_list = os.listdir(full_subdir)
        if len(img_list) < 2:
            continue
        img_dict[subdir] = img_list
        pair_count += int((len(img_list) * (len(img_list) - 1)) / 2)

    return pair_count, lambda : _yield_DFMsData_img(dataset_path, img_dict)


""" Load MapData dataset """

_MapData_PATH = r'E:\cmm_proj\dataset\MapData\MapData-test'

def _yield_MapData(dataset_path, count):
    for i in range(count):
        try:
            image_1 = imageio.imread(os.path.join(dataset_path, 'L', f'L{i+1}.png'))
            image_2 = imageio.imread(os.path.join(dataset_path, 'R', f'R{i+1}.png'))

            gt_file = open(os.path.join(dataset_path, 'GT', f'GT{i+1}.txt'), 'r', encoding='utf-8')
            gt_content = gt_file.readlines()
            gt_file.close()
            affine_matrix = []
            for line in gt_content:
                affine_matrix.append([float(i) for i in line.split()])
        except:
            image_1 = np.array([])
            image_2 = np.array([])
            affine_matrix = None

        yield f'pair_{i+1}', image_1, image_2, np.array(affine_matrix)

def load_MapData(dataset_path=_MapData_PATH):
    count = len(os.listdir(os.path.join(dataset_path, 'GT')))
    return count, lambda : _yield_MapData(dataset_path, count)
