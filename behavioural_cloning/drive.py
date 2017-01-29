import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = process_image(np.asarray(image))

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    a = float(model.predict(transformed_image_array, batch_size=1))
    steering_angle = a
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


def process_image(img):
    import cv2
    # img = cv2.resize(img, (160, 80))
    imshape = img.shape

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

    # sobelx = cv2.Sobel(combined_color, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(combined_color, cv2.CV_64F, 0, 1, ksize=3)
    # 3) Take the absolute value of the x and y gradients
    # x = np.absolute(sobelx)
    # y = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # g = np.arctan2(y, x)
    # 5) Create a binary mask where direction thresholds are met
    # mask = np.zeros_like(combined_color)
    # sobel_thresh = (0.3, 1.7)
    # mask[(g >= sobel_thresh[0]) & (g <= sobel_thresh[1])] = 1
    # visualize the sorbel image here if necessary
    # plt.imshow(mask, interpolation='nearest')
    # plt.show()

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
    hls[masked_image == 0.] = np.array([0., 0., 0.])

    return (np.array(hls).astype(np.float32) / 255.0) + 0.01


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())

    model.compile(loss='mean_squared_error', optimizer=Nadam())
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)