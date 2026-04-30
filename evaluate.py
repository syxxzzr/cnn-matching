import csv
import time
import os
import numpy as np
import cnnmatching
import load_dataset

_EVALUATE_EPSILON = 3.0
_DATASET_LOADER = load_dataset.load_OSdataset2

def evaluate_matches(matches, epsilon=3.0, affine_matrix=None):
    total = matches.shape[0]
    if total == 0:
        return {
            'NCM': 0, 'SR': 0.0, 'RMSE': 0.0
        }

    pts_1 = matches[:, 0, :]
    pts_2 = matches[:, 1, :]

    if affine_matrix is not None:
        affine = np.asarray(affine_matrix, dtype=float)
        if affine.shape == (2, 3):
            row = np.array([[0, 0, 1]], dtype=float)
            affine = np.vstack([affine, row])
        elif affine.shape != (3, 3):
            raise ValueError("affine_matrix must be (2,3) or (3,3)")
        ones = np.ones((total, 1))
        pts_1_h = np.hstack([pts_1, ones])
        pts_1 = (affine @ pts_1_h.T).T[:, :2]

    dists = np.linalg.norm(pts_1 - pts_2, axis=1)

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
    img_count, img_yield = _DATASET_LOADER()
    print(f'{img_count} image files found to calculate\n')


    result_dir = os.path.join('result', time.strftime('%Y-%m-%d %H-%M-%S'))
    os.makedirs(result_dir)

    csv_file = open(os.path.join(result_dir, 'results.csv'), mode='w', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['File Name', 'Total Matches', 'MT', 'NCM', 'SR', 'RMSE'])

    print('File Name        Total Matches MT(s)  NCM   SR     RMSE')
    print('-' * 16, '-' * 13, '-' * 5, '-' * 5, '-' * 6, '-' * 5)

    valid_count = 0
    indicators_sum = {'MT': 0., 'SR': 0., 'RMSE': 0.}

    for img_name, image_1, image_2, affine_matrix in img_yield():
        if image_1.shape == (0, ) or image_2.shape == (0, ):
            print(f'{img_name:<16} Failed to open file')
            continue

        start_time = time.perf_counter()
        matches = cnnmatching.extract_and_match(image_1, image_2)
        time_spend = time.perf_counter() - start_time

        res = evaluate_matches(matches, _EVALUATE_EPSILON, affine_matrix)

        csv_writer.writerow([img_name, matches.shape[0], time_spend, res['NCM'], res['SR'], res['RMSE']])
        np.savez(os.path.join(result_dir, f'{img_name}.npz'),
                 Matches=matches, MT=time_spend, NCM=res['NCM'], SR=res['SR'], RMSE=res['RMSE'])

        valid_count += 1
        indicators_sum['MT'] += time_spend
        indicators_sum['SR'] += res['SR']
        indicators_sum['RMSE'] += res['RMSE']

        print(f'{img_name:<16} {matches.shape[0]:<13d} {time_spend:<5.2f} '
              f'{res['NCM']:<5d} {res['SR']:<6.2%} {res['RMSE']:<5.2f}')

    print('-' * 55)
    if valid_count == 0:
        print(f'{'No valid image calculated':>55}')
    else:
        print('{:>55}'.format(
            f'Average:\tMT:{indicators_sum['MT'] / valid_count:<5.2f}s SR:{indicators_sum['SR'] / valid_count:<6.2%} RMSE:{indicators_sum['RMSE'] / valid_count:<5.2f}'))

    csv_file.close()
