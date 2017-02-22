import glob
import pickle
import numpy as np
import cv2
from sklearn.utils import shuffle


def duplicates(set1):
    dups = set([])
    norms = list(map(lambda x: np.tensordot(x, x, axes=3), set1))
    for i, x in enumerate(set1):
        if i in dups:
            continue
        if abs(norms[i]) <= 5:
            dups.add(i)
            continue
        for j, y in enumerate(set1):
            if i <= j:
                break
            if abs(norms[i] - norms[j]) > 5:
                continue
            t = np.tensordot(x, y, axes=3)
            if abs(t - norms[i]) < 5 and abs(t - norms[j]) < 5:
                dups.add(i)
                break
    print("{0} dups found. Process Complete".format(len(dups)))
    dups = list(dups)
    return np.delete(set1, dups, 0)

# Read in cars
images = glob.glob('vehicles_smallset/vehicles_smallset/cars1/*.jpeg')
cars = []
for image in images:
    cars.append(cv2.imread(image))
images = glob.glob('vehicles_smallset/vehicles_smallset/cars2/*.jpeg')
for image in images:
    cars.append(cv2.imread(image))
images = glob.glob('vehicles_smallset/vehicles_smallset/cars3/*.jpeg')
for image in images:
    cars.append(cv2.imread(image))
dataset = duplicates(cars)

# Read in  notcars
images = glob.glob('non-vehicles_smallset/non-vehicles_smallset/notcars1/*.jpeg')
notcars = []
for image in images:
    notcars.append(cv2.imread(image))
images = glob.glob('non-vehicles_smallset/non-vehicles_smallset/notcars2/*.jpeg')
for image in images:
    notcars.append(cv2.imread(image))
images = glob.glob('non-vehicles_smallset/non-vehicles_smallset/notcars3/*.jpeg')
for image in images:
    notcars.append(cv2.imread(image))
dataset2 = duplicates(notcars)

# use the smallest number of non duplicate data found
# to have even number of images from both sets
if len(dataset2) < len(dataset):
    dataset = dataset[:len(dataset2)]
else:
    dataset2 = dataset2[:len(dataset)]

# randomize and create training/test sets
rand_state = np.random.randint(0, 100)
features, labels = np.concatenate((dataset, dataset2)), [1 for i in dataset] + [0 for j in dataset2]
features, labels = shuffle(features, labels, random_state=rand_state)
X_test, y_test = features[:int(len(features) / 10)], labels[:int(len(features) / 10)]
features, labels = features[int(len(features) / 10):], labels[int(len(features) / 10):]

# save to pickle
with open('data.p', 'wb') as f:
    pickle.dump({'X_train': features, 'y_train': labels, 'X_test': X_test, 'y_test': y_test}, f)
