import csv
import time
import os
import numpy as np
import imageio.v2 as imageio
import cnnmatching

_TEST_IMG_1_PATH = r'E:\cmm_proj\QXSLAB_SAROPT\test\opt_256_oc_0.2'
_TEST_IMG_2_PATH = r'E:\cmm_proj\QXSLAB_SAROPT\test\sar_256_oc_0.2'


def evaluate_matches(matches, epsilon=3.0):
    total = matches.shape[0]
    if total == 0:
        return {
            'NCM': 0, 'SR': 0.0, 'RMSE': 0.0
        }

    dists = np.linalg.norm(matches[:, 0, :] - matches[:, 1, :], axis=1)

    correct_mask = dists <= epsilon
    NCM = np.sum(correct_mask)
    SR = NCM / total

    if NCM > 0:
        correct_dists = dists[correct_mask]
        RMSE = np.sqrt(np.mean(correct_dists ** 2))
    else:
        RMSE = 0.0

    return {
        'NCM': NCM,
        'SR': SR,
        'RMSE': RMSE,
    }


if __name__ == '__main__':
    # Only calculate the file both existed in the two folders
    img_list = list(set(os.listdir(_TEST_IMG_1_PATH)) & set(os.listdir(_TEST_IMG_2_PATH)))
    print(f'{len(img_list)} image files found to calculate\n')

    result_dir = os.path.join('result', time.strftime('%Y-%m-%d %H-%M-%S'))
    os.makedirs(result_dir)

    csv_file = open(os.path.join(result_dir, 'results.csv'), mode='w', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['File Name', 'Total Matches', 'MT', 'NCM', 'SR', 'RMSE'])

    print('File Name        Total Matches MT(s)  NCM   SR     RMSE')
    print('-'*16, '-'*13, '-'*5, '-'*5, '-'*6, '-'*5)

    for img_name in img_list:
        try:
            image_1 = imageio.imread(os.path.join(_TEST_IMG_1_PATH, img_name))
            image_2 = imageio.imread(os.path.join(_TEST_IMG_2_PATH, img_name))
        except:
            print(f'{img_name:<16d} Failed to open file')
            continue

        start_time = time.perf_counter()
        matches = cnnmatching.extract_and_match(image_1, image_2)
        time_spend = time.perf_counter() - start_time

        res = evaluate_matches(matches)

        csv_writer.writerow([img_name, matches.shape[0], time_spend, res['NCM'], res['SR'], res['RMSE']])
        np.savez(os.path.join(result_dir, f'{img_name}.npz'),
                 Matches=matches, MT=time_spend, NCM=res['NCM'], SR=res['SR'], RMSE=res['RMSE'])

        print(f'{img_name:<16} {matches.shape[0]:<13d} {time_spend:<5.2f} '
              f'{res['NCM']:<5d} {res['SR']:<6.2%} {res['RMSE']:<5.2f}')

    csv_file.close()
