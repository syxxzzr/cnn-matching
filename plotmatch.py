import numpy as np


def _as_3ch(image):
    if image.ndim == 2:
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image


def plot_matches(ax, image_1, image_2, matches,
                 keypoints_color='r', matches_color=None, plot_matche_points=True, matchline = True, matchlinewidth = 0.5,
                 alignment='horizontal'):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image_1 : (N, M [, 3]) array
        First grayscale or color image.
    image_2 : (N, M [, 3]) array
        Second grayscale or color image.
    matches : (N, 2, 2) array
        Keypoint coordinates as ``((row_1, col_1), (row_2, col_2))``.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.

    """

    #image1 = img_as_float(image1)
    #image2 = img_as_float(image2)
    image_1 = _as_3ch(image_1)
    image_2 = _as_3ch(image_2)

    new_shape1 = list(image_1.shape)
    new_shape2 = list(image_2.shape)

    if image_1.shape[0] < image_2.shape[0]:
        new_shape1[0] = image_2.shape[0]
    elif image_1.shape[0] > image_2.shape[0]:
        new_shape2[0] = image_1.shape[0]

    if image_1.shape[1] < image_2.shape[1]:
        new_shape1[1] = image_2.shape[1]
    elif image_1.shape[1] > image_2.shape[1]:
        new_shape2[1] = image_1.shape[1]

    if new_shape1 != image_1.shape:
        #new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1 = np.full(new_shape1, 255)
        new_image1[:image_1.shape[0], :image_1.shape[1]] = image_1
        image_1 = new_image1

    if new_shape2 != image_2.shape:
        #new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2 = np.full(new_shape2, 255)
        new_image2[:image_2.shape[0], :image_2.shape[1]] = image_2
        image_2 = new_image2

    offset = np.array(image_1.shape)
    if alignment == 'horizontal':
        if image_2.ndim == 3:
            blank = np.full((new_shape2[0], 10, 3), 255)
        if image_1.ndim == 2 or image_2.ndim == 2:
            blank = np.full((new_shape2[0], 10), 255)
        image = np.concatenate([image_1, blank, image_2], axis=1)
        offset[0] = 0
        offset[1] += 10
    elif alignment == 'vertical':
        if image_2.ndim == 3 :
            blank = np.full(10,(new_shape2[1], 3), 255)
        if image_1.ndim == 2:
            blank = np.full(10,(new_shape2[1]), 255)
        image = np.concatenate([image_1, blank, image_2], axis=0)
        offset[1] = 0
        offset[0] += 10
    else:
        mesg = ("plot_matches accepts either 'horizontal' or 'vertical' for "
                "alignment, but '{}' was given. See "
                "https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                "for details.").format(alignment)
        raise ValueError(mesg)


    if plot_matche_points:
        ax.scatter(matches[:, 0, 0], matches[:, 0, 1],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')
        ax.scatter(matches[:, 1, 0] + offset[1], matches[:, 1, 1] + offset[0],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, image_1.shape[1] + offset[1], image_1.shape[0] + offset[0], 0))

    if matchline == True:
        for match_point in matches:
            if matches_color is None:
                color = np.random.rand(3)
            else:
                color = matches_color

            ax.plot((match_point[0, 0], match_point[1, 0] + offset[1]),
                    (match_point[0, 1], match_point[1, 1] + offset[0]),
                    '-', color=color, linewidth=matchlinewidth, marker='+', markersize=8)


def plot_matches2(ax, image1, image2, keypoints1, keypoints2,
                 keypoints_color='r', matches_color=None, plot_matche_points=True, matchline = True, matchlinewidth = 0.5,
                 alignment='horizontal'):


    image1 = _as_3ch(image1)
    image2 = _as_3ch(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        #new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1 = np.full(new_shape1, 255)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        #new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2 = np.full(new_shape2, 255)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == 'horizontal':
        if image2.ndim == 3:
            blank = np.full((new_shape2[0], 10, 3), 255)
        if image1.ndim == 2 or image2.ndim == 2:
            blank = np.full((new_shape2[0], 10), 255)
        image = np.concatenate([image1, blank, image2], axis=1)
        offset[0] = 0
        offset[1] += 10
    elif alignment == 'vertical':
        if image2.ndim == 3 :
            blank = np.full(10,(new_shape2[1], 3), 255)
        if image1.ndim == 2:
            blank = np.full(10,(new_shape2[1]), 255)
        image = np.concatenate([image1, blank, image2], axis=0)
        offset[1] = 0
        offset[0] += 10
    else:
        mesg = ("plot_matches accepts either 'horizontal' or 'vertical' for "
                "alignment, but '{}' was given. See "
                "https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                "for details.").format(alignment)
        raise ValueError(mesg)


    if plot_matche_points:
        ax.scatter(keypoints1[:, 0], keypoints1[:, 1],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')
        ax.scatter(keypoints2[:, 0] + offset[1], keypoints2[:, 1] + offset[0],
                   facecolors='none', edgecolors=keypoints_color, marker = '.')

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    if matchline == True:
        for i in range(keypoints1.shape[0]):

            if matches_color is None:
                color = np.random.rand(3)
            else:
                color = matches_color

            ax.plot((keypoints1[i, 0], keypoints2[i, 0] + offset[1]),
                    (keypoints1[i, 1], keypoints2[i, 1] + offset[0]),
                    '-', color=color, linewidth=matchlinewidth, marker='+', markersize=5)



