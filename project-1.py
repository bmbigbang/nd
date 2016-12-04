#importing some useful packages
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # l2 = np.copy(lines)
    # line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    # draw_lines(line_img, l2)
    # plt.figure()
    # plt.imshow(line_img)

    lines = extrapolate(lines, img.shape)

    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

from numpy import polyfit


def extrapolate(lines, shape):
    vertices = np.array([[(imshape[1] / 6, imshape[0]),  # bottom left
                          ((imshape[1] - 40) / 2, (imshape[0] + 90) / 2),  # top left
                          ((imshape[1] + 50) / 2, (imshape[0] + 90) / 2),  # top right
                          (imshape[1] - (imshape[1] / 20), imshape[0])]], dtype=np.int32)  # bottom right
    x_mid = (shape[1]) / 2; x_max = shape[1]
    y_mid = (shape[0] + 90) / 2; y_max = shape[0]
    left_x = []; left_y = []; right_x = []; right_y = []
    for x0, y0, x1, y1 in [j[0] for j in lines]:
        if x0 < x_mid and x1 < x_mid:
            left_x.append(x0); left_x.append(x1)
            left_y.append(y0); left_y.append(y1)
        elif x0 > x_mid and x1 > x_mid:
            right_x.append(x0); right_x.append(x1)
            right_y.append(y0); right_y.append(y1)
    left = polyfit(left_x, left_y, deg=1)
    x0 = (imshape[0] - left[1]) / left[0]
    x1 = (((imshape[0] + 90) / 2) - left[1]) / left[0]
    new = [[x0, imshape[0], x1, (imshape[0] + 90) / 2]]

    right = polyfit(right_x, right_y, deg=1)
    x0 = (((imshape[0] + 90) / 2) - right[1]) / right[0]
    x1 = (imshape[0] - right[1]) / right[0]
    new += [[x0, (imshape[0] + 90) / 2, x1, imshape[0]]]
    return np.array([[i] for i in new], dtype='int32')


import os

for filename in os.listdir("test_images/"):
    orig_image = mpimg.imread('test_images/{}'.format(filename))
    image = (mpimg.imread('test_images/{}'.format(filename)) * 255).astype('uint8')

    gray = grayscale(image)

    kernel_size = 5
    blur_gray = gaussian_blur(image, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = image.shape
    vertices = np.array([[(imshape[1] / 6, imshape[0]),  # bottom left
                          ((imshape[1] - 40) / 2, (imshape[0] + 90) / 2),  # top left
                          ((imshape[1] + 50) / 2, (imshape[0] + 90) / 2),  # top right
                          (imshape[1] - (imshape[1] / 20), imshape[0])]], dtype=np.int32)  # bottom right
    # diagnose vertices
    # if i == 'solidWhiteCurve.jpg':
    #     mask = np.zeros_like(image)
    #     cv2.fillPoly(mask, vertices, 255)
    #     mask = cv2.bitwise_and(orig_image, mask)
    #     plt.figure()
    #     plt.imshow(mask)
    masked_image = region_of_interest(edges, vertices)

    rho = 0.7 # distance resolution in pixels of the Hough grid
    theta = np.pi / 360  # angular resolution in radians of the Hough grid
    threshold = 3  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10  # minimum number of pixels making up a line
    max_line_gap = 8  # maximum gap in pixels between connectable line segments
    lines = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)

    w_img = weighted_img(lines, orig_image)
    plt.figure()
    plt.xlabel('{}'.format(filename))
    plt.imshow(w_img)
    plt.imsave('test_images/Extrapolated-{}'.format(filename), w_img)

plt.show()