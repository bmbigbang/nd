import numpy as np
import matplotlib.pyplot as plt
import pandas

with open('behavioural_cloning/driving_log.csv', 'r') as f:
    t = pandas.read_csv(f, delimiter=',', sep='\s', header=None)
    labels_dict = {}

    for i in t.T.iteritems():
        if i[1][0].find('center'):
            labels_dict[i[1][0][i[1][0].find('center'):]] = i[1][3]


def process_images(n=[], labels=[]):
    import os
    for i in os.listdir('behavioural_cloning/IMG'):
        if not i.endswith('.jpg'):
            print(i)
            continue
        if not i.startswith('center'):
            continue
        # if abs(float(labels_dict[i])) < 0.05:
        #     labels_dict[i] = 0.05

        n.append(i)
        labels.append(labels_dict[i])

    return n, labels

features, labels = process_images()

# plt.imshow(features[np.random.randint(0, len(features))], interpolation='nearest')
# plt.show()
# plt.imshow(features[np.random.randint(0, len(features))], interpolation='nearest')
# plt.show()
# plt.imshow(features[np.random.randint(0, len(features))], interpolation='nearest')
# plt.show()
# plt.imshow(features[np.random.randint(0, len(features))], interpolation='nearest')
# plt.show()
# plt.imshow(features[np.random.randint(0, len(features))], interpolation='nearest')
# plt.show()
#
# import sys
# sys.exit(0)

from sklearn.utils import shuffle

X_valid, y_valid = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
features, labels = shuffle(features, labels, random_state=1)
X_test, y_test = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
image_shape = (160, 320, 3)
nb_epoch = 50
batch_size = 64

from random import randint
class generator:
    def __init__(self, X, y, batch_size=128, training=False):
        self.X = X
        self.y = np.float32(y)
        self.X, self.y = shuffle(self.X, self.y, random_state=1)
        # if training:
        #     temp = np.sort(np.dstack((np.exp(self.y), self.X, self.y)), axis=1)
        #     self.X = temp[0][::-1][:, 1]
        #     self.y = np.sign(np.array(temp[0][::-1][:, 2])) * np.array(temp[0][::-1][:, 0])
        # else:
        # self.y = np.sign(self.y) * np.exp(np.abs(self.y))
        # self.y[self.y == 0.] = np.float32(randint(0, 1) or -1)
        self.batch_size = batch_size
        self.step = 0

    def g(self):
        import cv2
        while True:
            feat, lab = [], np.array([])
            step = self.step * self.batch_size
            if not len(self.X[step:step + self.batch_size]):
                self.step = 0; step = 0
            else:
                self.step += 1
            for i, j in zip(self.X[step:step + self.batch_size], self.y[step:step + self.batch_size]):
                img = cv2.imread('behavioural_cloning/IMG/{}'.format(i))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_CUBIC)

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

                line_img = np.zeros(image_shape, dtype=np.uint8)
                color = [255, 0, 0]
                thickness = 1

                x_mid = imshape[1] / 2
                left_x = []; left_y = []; right_x = []; right_y = []
                for x0, y0, x1, y1 in [j[0] for j in lines]:
                    if x0 != x1 and x0 > x_mid and x1 > x_mid and 0 < ((y1 - y0) / (x1 - x0)) < 1:
                        left_x.append(x0); left_x.append(x1)
                        left_y.append(y0); left_y.append(y1)
                    elif x0 != x1 and x0 < x_mid and x1 < x_mid and 0 > ((y1 - y0) / (x1 - x0)) > -1:
                        right_x.append(x0); right_x.append(x1)
                        right_y.append(y0); right_y.append(y1)

                if not left_x or not left_y:
                    for line in lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(masked_image, (x1, y1), (x2, y2), color, thickness)

                    plt.imshow(masked_image, interpolation='nearest')
                    plt.show()
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

                    plt.imshow(masked_image, interpolation='nearest')
                    plt.show()
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

                # visualise the individual processed images here if necessary
                # plt.imshow(img, interpolation='nearest')
                # plt.show()
                feat.append(img)
                lab = np.append(lab, j)

            # normalize by diving by (maximum - minimum) after subtracting minimum
            # add a small constant (0.01) to shift away from 0 for better performance operations
            yield (np.array(feat).astype(np.float32) / 255.0) + 0.01, lab


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from keras.optimizers import Nadam


model = Sequential()
model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), input_shape=image_shape,
                 init='normal', border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, init='normal', subsample=(2, 2),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, init='normal', subsample=(2, 2),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='normal', subsample=(1, 1),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='normal', subsample=(1, 1),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Flatten())
model.add(Dense(1164, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mean_squared_error', optimizer=Nadam())

for epoch in range(nb_epoch):
    gen = generator(features, labels, batch_size=batch_size, training=True).g()
    test_gen = generator(X_test, y_test, batch_size=batch_size).g()
    valid_gen = generator(X_valid, y_valid, batch_size=batch_size).g()

    training_loss = []
    for st in range(len(features) % batch_size):
        X, y = next(gen)
        metrics = model.train_on_batch(X, y)
        training_loss.append(metrics)

    st += 1
    print('Epoch {}  Loss: {}'.format(epoch + 1, sum(training_loss) / st))

    testing_loss = []
    for st in range(len(X_test) % batch_size):
        X_t, y_t = next(test_gen)
        _ = model.test_on_batch(X_t, y_t)
        testing_loss.append(_)
    st += 1
    print('Testing Loss: {}'.format(sum(testing_loss) / st))

    valid_loss = []
    for step in range(len(X_valid) % batch_size):
        X_v, y_v = next(valid_gen)
        _ = model.test_on_batch(X_v, y_v)
        valid_loss.append(_)
    st += 1
    print('Validation Loss: {}'.format(sum(valid_loss) / st))

with open('behavioural_cloning/model.json', 'w+') as f:
    f.write(model.to_json())
model.save_weights('behavioural_cloning/model.h5')




