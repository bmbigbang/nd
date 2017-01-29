from keras.models import model_from_json
from keras.optimizers import Nadam
import matplotlib.pyplot as plt
import numpy as np
import cv2


def process_image(img):
    img = cv2.resize(img, (160, 80))
    imshape = img.shape
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # visualize the colour thresholding image here if necessary
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()

    shape_mask = np.zeros_like(hls)
    # vertices are of two trapezoids with the most significant parts of the road
    vertices = np.array([
        [(((imshape[1] - 30) / 2), ((imshape[0] + 20) / 2)),
         (40, imshape[0]),
         (0, imshape[0]),  # bottom left
         (0, (imshape[0] - 30) / 2),  # mid left
         ((imshape[1] - 70) / 2, (imshape[0] - 20) / 2),  # top left
         ((imshape[1] + 70) / 2, (imshape[0] - 20) / 2),  # top right
         (imshape[1], (imshape[0] - 30) / 2),  # mid right
         (imshape[1], imshape[0]),  # bottom right
         (imshape[1] - 40, imshape[0]),
         (((imshape[1] + 30) / 2), ((imshape[0] + 20) / 2))]
    ], dtype=np.int32)
    cv2.fillPoly(shape_mask, vertices, (255, 255, 255))

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(hls, shape_mask)

    return (np.array(masked_image).astype(np.float32) / 255.0) + 0.01


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


