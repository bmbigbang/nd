from keras.models import model_from_json
from keras.optimizers import Nadam
import matplotlib.pyplot as plt
import numpy as np
import cv2


def process_image(img):
    # img = cv2.resize(img, (160, 80))

    imshape = img.shape
    plt.imshow(img)
    plt.show()
    # use gaussian blurring for better matching of edges to actual lines than artifacts
    R = img[:, :, 2]
    R_thresh = (200, 255)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls[:, :, 2]
    H = hls[:, :, 0]
    S_thresh = (90, 255)
    H_thresh = (15, 100)
    mask = np.zeros_like(S)
    mask[((S > S_thresh[0]) & (S <= S_thresh[1])) |
         ((R > R_thresh[0]) & (R <= R_thresh[1])) |
         ((H > H_thresh[0]) & (H <= H_thresh[1]))] = 1
    combined_color = mask
    # visualize the colour thresholding image here if necessary
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()

    sobelx = cv2.Sobel(combined_color, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(combined_color, cv2.CV_64F, 0, 1, ksize=3)
    # 3) Take the absolute value of the x and y gradients
    x = np.absolute(sobelx)
    y = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    g = np.arctan2(y, x)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(combined_color)
    sobel_thresh = (0.5, 1.5)
    mask[(g >= sobel_thresh[0]) & (g <= sobel_thresh[1])] = 1
    # visualize the sorbel image here if necessary
    plt.imshow(mask, interpolation='nearest')
    plt.show()

    shape_mask = np.zeros_like(mask)
    # vertices are of two trapezoids with the most significant parts of the road
    vertices = np.array([
        [(((imshape[1] - 60) / 2), ((imshape[0] + 40) / 2)),
         (50, imshape[0]),
         (0, imshape[0]),  # bottom left
         (0, (imshape[0] - 30) / 2),  # mid left
         ((imshape[1] - 140) / 2, (imshape[0] - 40) / 2),  # top left
         ((imshape[1] + 140) / 2, (imshape[0] - 40) / 2),  # top right
         (imshape[1], (imshape[0] - 30) / 2),  # mid right
         (imshape[1], imshape[0]),  # bottom right
         (imshape[1] - 50, imshape[0]),
         (((imshape[1] + 60) / 2), ((imshape[0] + 40) / 2))]
    ], dtype=np.int32)
    cv2.fillPoly(shape_mask, vertices, (255, 255, 255))

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(mask, shape_mask)
    hls[masked_image == 0.] = np.array([0.01, 0.01, 0.01])
    # visualise the individual processed images here if necessary
    plt.imshow(masked_image, interpolation='nearest')
    plt.show()

    return (np.array(hls).astype(np.float32) / 255.0) + 0.01


img = cv2.imread('IMG2/center_2017_01_28_19_07_56_423.jpg')
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


