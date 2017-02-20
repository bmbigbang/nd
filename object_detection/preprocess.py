import glob
import pickle
import numpy as np
import cv2


def duplicates(set1):
    s = len(set1)
    dups = set([])
    norms1 = map(lambda x: np.tensordot(x, x, axes=3), set1)
    candidates = near_norms(norms1, s)
    norms1 = list(norms1)
    print("starting direct comparisons with {0} dup candidates out of {1}".format(len(candidates), s))
    s = float(len(candidates))
    for i, x in enumerate(candidates):
        if i % 200 == 0:
            print("{0:.2f}%".format(100 * i / s))
        if x in dups:
            continue
        if abs(norms1[x]) <= 1e-7:
            dups.add(x)
            continue
        for j in candidates:
            if x <= j:
                break
            t = np.tensordot(set1[x], set1[j], axes=2)
            if (t / norms1[x]) > 0.948 and (t / norms1[j]) > 0.948:
                dups.add(x)
                break
    print("{0} dups found. Process Complete".format(len(dups)))
    dups = list(dups)
    return np.delete(set1, dups, 0)


def near_norms(s, total):
    cands = set()
    print("starting norm candidates")
    for i, x in enumerate(s):
        if i % 2000 == 0:
            print("{0:.2f}%".format(100 * i / total))
        for j, y in enumerate(s):
            if i <= j:
                break
            if abs(x - y) <= 5e-5:
                cands.add(i)
                cands.add(j)
    print("100% completed norm candidates")
    return sorted(list(cands))

# Read in cars
images = glob.glob('vehicles_smallset/vehicles_smallset/cars1/*.jpeg')
cars = []
for image in images:
    cars.append(cv2.imread(image))
dataset = duplicates(cars)

# Read in  notcars
images = glob.glob('non-vehicles_smallset/non-vehicles_smallset/notcars1/*.jpeg')
notcars = []
for image in images:
    notcars.append(cv2.imread(image))
dataset2 = duplicates(notcars)
