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
    a = np.float32(model.predict(transformed_image_array, batch_size=1))
    if np.abs(a) <= 1:
        steering_angle = 0
    else:
        steering_angle = np.sign(a) * np.log(np.abs(a))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


def process_image(img):
    import cv2

    imshape = img.shape
    mask = np.zeros_like(img)
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

    kernel_size = 3
    blur_gray = cv2.GaussianBlur(masked_image, (kernel_size, kernel_size), 0)

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

    line_img = np.zeros(imshape, dtype=np.uint8)
    color = [255, 0, 0]
    thickness = 1

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

    if not left_x or not left_y:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(masked_image, (x1, y1), (x2, y2), color, thickness)

        raise Exception('Empty left line image array')
    else:
        left = np.polyfit(left_x, left_y, deg=1)
        x0 = (imshape[0] - left[1]) / left[0]
        x1 = (((imshape[0] - 40) / 2) - left[1]) / left[0]
        new = [[x1, ((imshape[0] - 40) / 2), x0, imshape[0]]]

    if not right_x or not right_y:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(masked_image, (x1, y1), (x2, y2), color, thickness)

        raise Exception('Empty right line image array')
    else:
        right = np.polyfit(right_x, right_y, deg=1)
        x0 = (((imshape[0] - 40) / 2) - right[1]) / right[0]
        x1 = (imshape[0] - right[1]) / right[0]
        new += [[x0, (imshape[0] - 40) / 2, x1, imshape[0]]]
    # np.array([[iii] for iii in new], dtype='int32')
    for line in np.array([[iii] for iii in new], dtype='int32'):
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # plt.imshow(img, interpolation='nearest')
    # plt.show()
    return img


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


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)