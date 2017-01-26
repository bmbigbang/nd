from keras.models import model_from_json
from keras.optimizers import Nadam
import numpy as np
import cv2


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 80))

    imshape = img.shape
    mask = np.zeros_like(img)
    # vertices are of two trapezoids with the most significant parts of the road
    vertices = np.array([
        [(((imshape[1] - 80) / 2), ((imshape[0] + 20) / 2)),
         (0, imshape[0]),  # bottom left
         (0, (imshape[0] * 6 / 12)),  # mid left
         ((imshape[1] - 100) / 2, (imshape[0] - 40) / 2),  # top left
         ((imshape[1] + 100) / 2, (imshape[0] - 40) / 2),  # top right
         (imshape[1], (imshape[0] * 7 / 12)),  # mid right
         (imshape[1], imshape[0]),  # bottom right
         (imshape[1], imshape[0]),
         (((imshape[1] + 60) / 2), ((imshape[0] + 20) / 2))]
    ], dtype=np.int32)
    cv2.fillPoly(mask, vertices, (255, 255, 255))

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    # use gaussian blurring for better matching of edges to actual lines than artifacts
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(masked_image, (kernel_size, kernel_size), 0)

    # set threshholds and find canny edges
    low_threshold = 20
    high_threshold = 200
    canny = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 0.5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 450  # angular resolution in radians of the Hough grid
    threshold = 3  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 8  # minimum number of pixels making up a line
    max_line_gap = 4  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # set line parameters
    color = [255, 0, 0]
    thickness = 3

    # look at the left and right sides to plot a linear line
    x_mid = imshape[1] / 2
    left_x = [];
    left_y = [];
    right_x = [];
    right_y = []
    for x0, y0, x1, y1 in [j[0] for j in lines]:
        if x0 != x1 and x0 > x_mid and x1 > x_mid and 0 < ((y1 - y0) / (x1 - x0)) < 1:
            left_x.append(x0);
            left_x.append(x1)
            left_y.append(y0);
            left_y.append(y1)
        elif x0 != x1 and x0 < x_mid and x1 < x_mid and 0 > ((y1 - y0) / (x1 - x0)) > -1:
            right_x.append(x0);
            right_x.append(x1)
            right_y.append(y0);
            right_y.append(y1)

    # fit to the found lines, must have found some lines or this will fail!

    # left = np.polyfit(left_x, left_y, deg=1)
    # x0 = (imshape[0] - left[1]) / left[0]
    # x1 = (((imshape[0] - 40) / 2) - left[1]) / left[0]
    # new = [[x1, ((imshape[0] - 40) / 2), x0, imshape[0]]]
    #
    # right = np.polyfit(right_x, right_y, deg=1)
    # x0 = (((imshape[0] - 40) / 2) - right[1]) / right[0]
    # x1 = (imshape[0] - right[1]) / right[0]
    # new += [[x0, (imshape[0] - 40) / 2, x1, imshape[0]]]
    #
    # # plot the two lines
    # for line in np.array([[iii] for iii in new], dtype='int32'):
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    return (np.array(img).astype(np.float32) / 255.0) + 0.01


img = cv2.imread('IMG/center_2016_12_01_13_36_53_612.jpg')
image_array = process_image(np.asarray(img))

transformed_image_array = image_array[None, :, :, :]
# This model currently assumes that the features of the model are just the
with open('model.json', 'r') as jfile:
    # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
    # then you will have to call:
    #
    #   model = model_from_json(json.loads(jfile.read()))\
    #
    # instead.
    model = model_from_json(jfile.read())

model.compile(loss='mean_squared_error', optimizer=Nadam())
weights_file = 'model.h5'
model.load_weights(weights_file)
print(model.predict(transformed_image_array))


