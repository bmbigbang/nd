import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import os
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from object_detection.lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


# load samples from preprocessed pickle
with open('data.p', 'rb') as f:
    p = pickle.load(f)

cars = p['cars']
notcars = p['notcars']

### TODO: Tweak these parameters and see how the results change.
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()


# Define a class to retain the heatmaps within the last n frames
class Heatmap:
    def __init__(self, n=6):
        # number of images to average over
        self.n = n
        # keep the last heatmaps in memory
        self.heatmap = []


def process_image(image):
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        draw_image = cv2.cvtColor(image, getattr(cv2, 'COLOR_RGB2{}'.format(color_space)))
    else:
        draw_image = np.copy(image)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)

    # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                        xy_window=window_size, xy_overlap=overlap)
    ystart = 400
    ystop = 656

    hot_windows = []
    for scale in (1.2, 1.3, 1.4, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7):
        hot_windows.extend(find_cars(draw_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                     cell_per_block, spatial_size, hist_bins))

    # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
    #                              spatial_size=spatial_size, hist_bins=hist_bins,
    #                              orient=orient, pix_per_cell=pix_per_cell,
    #                              cell_per_block=cell_per_block, hog_channel=hog_channel,
    #                              spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    for box in hot_windows:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # store the heatmap and get the average before applying threshold
    stored.heatmap.append(heatmap)
    if len(stored.heatmap) > stored.n:
        stored.heatmap.pop(0)
    heatmap = np.add.reduce(stored.heatmap, 0) / len(stored.heatmap)

    # Zero out pixels below the threshold of 3
    heatmap[heatmap <= 3] = 0

    # visualise the thresholded heatmap here if necessary
    # plt.imshow(heatmap)
    # plt.show()

    labels = label(heatmap)

    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)

    # plt.imshow(image)
    # plt.show()
    return image

for i in os.listdir('test_images'):
    if not i.endswith(('jpg', 'png')):
        continue
    image = cv2.imread(r'test_images/{}'.format(i))
    image = cv2.cvtColor(image, getattr(cv2, 'COLOR_BGR2RGB'))
    stored = Heatmap()
    cv2.imwrite('output_images/{}'.format(i), process_image(image))


from moviepy.editor import VideoFileClip
stored = Heatmap(n=8)
output = 'processed_project_video.mp4'
clip2 = VideoFileClip('project_video.mp4')
clip = clip2.fl_image(process_image)
clip.write_videofile(output, audio=False)