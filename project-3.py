import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Dropout
from keras.optimizers import Nadam

with open('behavioural_cloning/driving_log.csv', 'r') as f:
    t = pandas.read_csv(f, delimiter=',', sep='\s', header=None)
    labels_dict = {}
    # compile the labels from the csv file into a dict with the file name as keys
    for i in t.T.iteritems():
        if i[1][0].find('center'):
            labels_dict[i[1][0][i[1][0].find('center'):]] = i[1][3]


def process_images(n=[], labels=[]):
    # loop through the folder and compile the labels read with the file names
    for i in os.listdir('behavioural_cloning/IMG'):
        # in case there are other files in this folder
        if not i.endswith('.jpg'):
            print(i)
            continue
        # pick center only files
        if not i.startswith('center'):
            continue
        # discard near zero values
        # if abs(float(labels_dict[i])) * 180/3.14 < 0.05:
        #     continue
        n.append(i)
        labels.append(labels_dict[i])

    return n, labels

features, labels = process_images()
# the first 1/10 of the images are set as the validation set. this set is always the same
# and is never shuffled
X_valid, y_valid = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
# shuffle and set another 1/10 to test set
features, labels = shuffle(features, labels, random_state=0)
X_test, y_test = features[:int(len(features)/10)], labels[:int(len(features)/10)]
features, labels = features[int(len(features)/10):], labels[int(len(features)/10):]
# set image and batch processing parameters
image_shape = (160, 320, 3)
nb_epoch = 10
batch_size = 128


class Generator:
    def __init__(self, X, y, batch_size=128):
        # read in the filenames and corresponding labels as features->X, labels->y
        self.X = X
        self.y = np.float32(y)
        # shuffle upon class initalization, happens at each epoch
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        # set batch size dynamically
        self.batch_size = batch_size
        self.step = 0

    def g(self):
        # depending on the cudnn version and opencv2, sometimes cv2 would crash for me
        # therefore i am importing locally so that it does not conflict with global scope variables
        import cv2
        while True:
            feat, lab = [], np.array([])
            # calculate the step based on batch size and reset to zero if the end of the files is reached
            step = self.step * self.batch_size
            if not len(self.X[step:step + self.batch_size]):
                self.step = 0; step = 0
            else:
                self.step += 1
            for i, j in zip(self.X[step:step + self.batch_size], self.y[step:step + self.batch_size]):
                # read the image and convert colours
                img = cv2.imread('behavioural_cloning/IMG/{}'.format(i))

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
                # visualise the individual processed images here if necessary
                # plt.imshow(hls, interpolation='nearest')
                # plt.show()
                feat.append((hls.astype(np.float32) / 255.0) + 0.01)
                lab = np.append(lab, j)

            # normalize by diving by (maximum - minimum) after subtracting minimum
            # add a small constant (0.01) to shift away from 0 for better performance operations
            yield np.array(feat), lab


# implement the nvidia self driving car neural network
model = Sequential()
model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), input_shape=image_shape,
                 init='normal', border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=28, nb_row=5, nb_col=5, init='normal', subsample=(2, 2),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=32, nb_row=5, nb_col=5, init='normal', subsample=(2, 2),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=52, nb_row=3, nb_col=3, init='normal', subsample=(1, 1),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Dropout(0.5))
model.add(Conv2D(nb_filter=52, nb_row=3, nb_col=3, init='normal', subsample=(1, 1),
                 border_mode='valid', dim_ordering='tf', activation='linear'))
model.add(Flatten())
model.add(Dense(1114, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()
# use Nadam() for enhanced learning rate of small image set
model.compile(loss='mean_squared_error', optimizer=Nadam())

for epoch in range(nb_epoch):
    # initialize the generators for each epoch, this randomizes the individual data sets
    gen = Generator(features, labels, batch_size=batch_size).g()
    test_gen = Generator(X_test, y_test, batch_size=batch_size).g()
    valid_gen = Generator(X_valid, y_valid, batch_size=batch_size).g()

    training_loss = []
    for st in range(len(features) % batch_size):
        # grab next batch and train. record the loss to report
        X, y = next(gen)
        metrics = model.train_on_batch(X, y)
        training_loss.append(metrics)

    st += 1
    print('Epoch {}  Loss: {}'.format(epoch + 1, sum(training_loss) / st))

    testing_loss = []
    for st in range(len(X_test) % batch_size):
        # grab next batch and test. record the loss to report
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

# save files
with open('behavioural_cloning/model.json', 'w+') as f:
    f.write(model.to_json())
model.save_weights('behavioural_cloning/model.h5')
