import glob
import pickle
import numpy as np
import cv2
from sklearn.utils import shuffle


def duplicates(set1):
    # initialize set to store duplicate values
    dups = set([])
    # create a map of vector lengths in three colour space
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
            if abs(t - norms[i]) < 3 and abs(t - norms[j]) < 3:
                dups.add(i)
                break
    print("{0} dups found. Process Complete".format(len(dups)))
    dups = list(dups)
    return np.delete(set1, dups, 0)

# Read in cars
images = glob.glob('vehicles/vehicles/GTI_Far/*.png')
cars = []
for image in images:
    cars.append(cv2.imread(image))
images = glob.glob('vehicles/vehiclest/GTI_Left/*.png')
for image in images:
    cars.append(cv2.imread(image))
images = glob.glob('vehicles/vehicles/GTI_MiddleClose/*.png')
for image in images:
    cars.append(cv2.imread(image))
images = glob.glob('vehicles/vehicles/GTI_Right/*.png')
for image in images:
    cars.append(cv2.imread(image))
dataset = duplicates(cars)

# Read in  notcars
images = glob.glob('non-vehicles/non-vehicles/Extras/*.png')
notcars = []
for image in images:
    notcars.append(cv2.imread(image))
images = glob.glob('non-vehicles/non-vehicles/GTI/*.png')
for image in images:
    notcars.append(cv2.imread(image))
dataset2 = duplicates(notcars)

# use the smallest number of non duplicate data found
# to have even number of images from both sets
if len(dataset2) < len(dataset):
    dataset = dataset[:len(dataset2)]
else:
    dataset2 = dataset2[:len(dataset)]

# save to pickle
with open('data.p', 'wb') as f:
    pickle.dump({'cars': dataset, 'notcars': dataset2}, f)
