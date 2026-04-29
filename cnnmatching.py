import time
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, transform

import plotmatch
from lib.cnn_feature import cnn_feature_extract

_RESIDUAL_THRESHOLD = 30
_DEFAULT_IMAGE_1 = "df-ms-data/1/df-uav-sar-500.jpg"
_DEFAULT_IMAGE_2 = "df-ms-data/1/df-googleearth-1k-20181029.jpg"


def _extract_and_match(image_1, image_2):
    kps_1, _, des_1 = cnn_feature_extract(image_1, nfeatures=-1)
    kps_2, _, des_2 = cnn_feature_extract(image_2, nfeatures=-1)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=40),
    )
    knn_matches = flann.knnMatch(des_1, des_2, k=2)

    good_matches = []
    locations_1_to_use = []
    locations_2_to_use = []

    distance_diff_avg = 0
    for match_1, match_2 in knn_matches:
        distance_diff_avg += match_2.distance - match_1.distance
    distance_diff_avg = distance_diff_avg / len(knn_matches)

    for match_1, match_2 in knn_matches:
        if match_2.distance > match_1.distance + distance_diff_avg:
            good_matches.append(match_1)
            point_2 = cv2.KeyPoint(
                kps_2[match_1.trainIdx][0],
                kps_2[match_1.trainIdx][1],
                1,
            )
            point_1 = cv2.KeyPoint(
                kps_1[match_1.queryIdx][0],
                kps_1[match_1.queryIdx][1],
                1,
            )
            locations_1_to_use.append([point_1.pt[0], point_1.pt[1]])
            locations_2_to_use.append([point_2.pt[0], point_2.pt[1]])

    print("match num is %d" % len(good_matches))

    locations_1_to_use = np.array(locations_1_to_use)
    locations_2_to_use = np.array(locations_2_to_use)

    _, inliers = measure.ransac(
        (locations_1_to_use, locations_2_to_use),
        transform.AffineTransform,
        min_samples=3,
        residual_threshold=_RESIDUAL_THRESHOLD,
        max_trials=1000,
    )
    matches = np.stack((locations_1_to_use[inliers], locations_2_to_use[inliers]), axis=1)

    return matches


def _show_matches(image1, image2, matches):
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.figsize"] = (4.0, 3.0)

    _, axis = plt.subplots()
    plotmatch.plot_matches(
        axis,
        image1,
        image2,
        matches,
        plot_matche_points=False,
        matchline=True,
        matchlinewidth=0.3,
    )
    axis.axis("off")
    axis.set_title("")
    plt.show()


def main(imgfile1=_DEFAULT_IMAGE_1, imgfile2=_DEFAULT_IMAGE_2):
    start = time.perf_counter()
    image1 = imageio.imread(imgfile1)
    image2 = imageio.imread(imgfile2)
    print("read image time is %6.3f" % (time.perf_counter() - start))

    start = time.perf_counter()
    matches = _extract_and_match(image1, image2)
    print("whole time is %6.3f" % (time.perf_counter() - start))

    _show_matches(image1, image2, matches)


if __name__ == "__main__":
    main()
