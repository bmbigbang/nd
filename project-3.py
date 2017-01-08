import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
features, labels = shuffle(features, labels, random_state=0)
X_test, y_test = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
image_shape = (160, 320, 3)
nb_epoch = 10
batch_size = 128


class generator:
    def __init__(self, X, y, batch_size=128):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.step = 0

    def g(self):
        import cv2
        while True:
            feat, lab = [], []
            step = self.step * self.batch_size
            if not self.X[step:step + self.batch_size]:
                self.step = 0; step = 0
            else:
                self.step += 1
            for i, j in zip(self.X[step:step + self.batch_size], self.y[step:step + self.batch_size]):

                img = cv2.imread('behavioural_cloning/IMG/{}'.format(i))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                kernel_size = 3
                blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

                low_threshold = 30
                high_threshold = 220
                canny = cv2.Canny(blur_gray, low_threshold, high_threshold)

                rho = 0.7  # distance resolution in pixels of the Hough grid
                theta = np.pi / 450  # angular resolution in radians of the Hough grid
                threshold = 2  # minimum number of votes (intersections in Hough grid cell)
                min_line_len = 7  # minimum number of pixels making up a line
                max_line_gap = 7  # maximum gap in pixels between connectable line segments
                lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                        maxLineGap=max_line_gap)

                line_img = np.zeros(img.shape, dtype=np.uint8)
                imshape = img.shape
                color = [255, 0, 0]
                thickness = 1
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if x1 > (imshape[1] + 120) / 2 and x2 > (imshape[1] + 120) / 2:

                            if x2 != x1 and 0.75 < ((y2 - y1) / (x2 - x1)) < 1.25:
                                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                        elif x1 < (imshape[1] - 120) / 2 and x2 < (imshape[1] - 120) / 2:
                            if x2 != x1 and -0.75 > ((y2 - y1) / (x2 - x1)) > -1.25:
                                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

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
                masked_image = cv2.bitwise_and(line_img, mask)
                # plt.imshow(masked_image, interpolation='nearest')
                # plt.show()
                feat.append(masked_image), lab.append(np.float32(j))

            yield np.array(feat).astype(np.float32) + 0.01, np.array(lab)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation
from keras.layers import Dropout
from keras.optimizers import SGD

# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     featurewise_std_normalization=False,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(nb_filter=6, nb_row=5, nb_col=5, input_shape=image_shape, init='normal',
                 border_mode='valid', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Conv2D(nb_filter=16, nb_row=7, nb_col=7, init='normal',
                 border_mode='valid', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Conv2D(nb_filter=28, nb_row=5, nb_col=5, init='normal',
                 border_mode='valid', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Conv2D(nb_filter=42, nb_row=5, nb_col=5, init='normal',
                 border_mode='valid', activation='relu', dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, init='normal',  activation='relu'))
model.add(Dense(1, init='normal', activation='relu'))
model.add(Activation('tanh'))
# model.add(Activation('softmax'))
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
# history = model.fit(X_train, y_train,
#                     batch_size=128, nb_epoch=nb_epoch,
#                     verbose=1, validation_data=(X_valid, y_valid))

gen = generator(features, labels, batch_size=batch_size).g()
test_gen = generator(X_test, y_test, batch_size=batch_size).g()
for epoch in range(nb_epoch):
    X, y = next(gen)
    X_t, y_t = next(test_gen)
    model.train_on_batch(X, y)
    model.test_on_batch(X_t, y_t)
# model.fit_generator(generator(features, labels, batch_size=batch_size).g(), int(len(features)/nb_epoch),
#                     nb_epoch, nb_worker=1, verbose=2, callbacks=[],
#                     validation_data=generator(X_valid, y_valid, batch_size=batch_size).g(),
#                     nb_val_samples=len(X_valid))



