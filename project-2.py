# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(train['features'])

# TODO: Number of testing examples.
n_test = len(test['features'])

# TODO: What's the shape of an traffic sign image?
image_shape = train['features'][0].shape
num_channels = 1  # grey scale

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(train['labels']) - min(train['labels']) + 1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Visualizations will be shown in the notebook.
# %matplotlib inline
plt.imshow(train['features'][np.random.randint(0, len(train['features']))], interpolation='nearest', cmap=cm.brg)

from collections import Counter
with open('signnames.csv', 'r') as f:
    temp = f.readlines()
    sign_map = {int(i.split(",")[0]): i.split(",")[1].strip() for i in temp if not i.startswith('ClassId')}
a = [[], [], []]; b = [[], []]
for i in Counter(train['labels']).items():
    if len(a[0]) < 15:
        a[0].append([sign_map[i[0]], i[1]])
    elif len(a[1]) < 15:
        a[1].append([sign_map[i[0]], i[1]])
    else:
        a[2].append([sign_map[i[0]], i[1]])
    b[0].append(i[0]); b[1].append(i[1])
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 9))
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].table(cellText=a[0], colLabels=['Sign', 'Count'], loc='center', fontsize=13).auto_set_font_size(False)
ax[1].table(cellText=a[1], colLabels=['Sign', 'Count'], loc='center', fontsize=13).auto_set_font_size(False)
ax[2].table(cellText=a[2], colLabels=['Sign', 'Count'], loc='center', fontsize=13).auto_set_font_size(False)
# plt.plot(b[0], b[1], label='Count of labels vs label id')

# plt.show()


### Step 1. Preprocess the data here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from sklearn.utils import shuffle
# create validation set
X_valid, X_test = np.split(X_test, 2)
y_valid, y_test = np.split(y_test, 2)
# randomize training/test data
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)
# reformat to grayscale
temp = np.zeros_like(X_train, dtype=np.float32)
temp = temp.reshape((-1, image_shape[0], image_shape[0], num_channels))
for i, j in enumerate(X_train):
    for k, l in enumerate(j):
        temp[i][k] = np.array([np.mean(l)])
temp = temp[:i + 1,:,:]
X_train = temp

temp = np.zeros_like(X_valid, dtype=np.float32)
temp = temp.reshape((-1, image_shape[0], image_shape[0], num_channels))
for i, j in enumerate(X_valid):
    for k, l in enumerate(j):
        temp[i][k] = np.array([np.mean(l)])
temp = temp[:i + 1,:,:]
X_valid = temp

temp = np.zeros_like(X_test, dtype=np.float32)
temp = temp.reshape((-1, image_shape[0], image_shape[0], num_channels))
for i, j in enumerate(X_test):
    for k, l in enumerate(j):
        temp[i][k] = np.array([np.mean(l)])
temp = temp[:i + 1,:,:]
X_test = temp


# normalize training data
X_train = ((X_train - X_train.min()) * (X_train.min() / X_train.max())) / (X_train.max() - X_train.min())

X_test = ((X_test - X_test.min()) * (X_test.min() / X_test.max())) / (X_test.max() - X_test.min())

X_valid = ((X_valid - X_valid.min()) * (X_valid.min() / X_valid.max())) / (X_valid.max() - X_valid.min())

### Q1. Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

### Q2. Define your architecture here.
### Feel free to use as many code cells as needed.

from tensorflow.contrib.layers import flatten

batch_size = 32
filter_size = 5
depth = 6
depth2 = 16


def model(data):
    layer1_weights = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, num_channels, depth], mean=0, stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros(depth))
    conv1 = tf.nn.conv2d(data, filter=layer1_weights, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv1 = tf.nn.relu(conv1 + layer1_biases)

    layer2_weights = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, depth, depth2], mean=0, stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros(depth2))
    conv2 = tf.nn.conv2d(conv1, filter=layer2_weights, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.relu(conv2 + layer2_biases)

    fc0 = flatten(conv2)
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.nn.relu(tf.matmul(fc0, fc1_W) + fc1_b)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes),  mean=0, stddev=0.1))
    fc3_b = tf.Variable(tf.zeros(n_classes))

    keep_prob = tf.Variable(0.5)  # dropout layer
    return tf.nn.dropout(tf.matmul(fc2, fc3_W) + fc3_b, keep_prob)



# Variables and Input data.
tf_train_dataset = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], num_channels))
tf_train_labels = tf.placeholder(tf.int32, shape=(None))

# Training computation.
one_hot_y = tf.one_hot(tf_train_labels, n_classes)
logits = model(tf_train_dataset)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y))

# introducing variable learning rate
global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.train.exponential_decay(0.01, global_step, 60000, 0.99)
# Optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss, global_step=global_step)
# Predictions for the training, validation, and test data.
# train_prediction = tf.nn.softmax(logits)
# valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
# test_prediction = tf.nn.softmax(model(tf_test_dataset))

### Q3. Train your model here.
### Feel free to use as many code cells as needed.
EPOCHS = 1000

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={tf_train_dataset: batch_x, tf_train_labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={tf_train_dataset: batch_x, tf_train_labels: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

# try:
#     saver
# except NameError:
#     saver = tf.train.Saver()
# saver.save(sess, 'lenet')
# print("Model saved")

### Step 2. Load the images and plot them here.
### Feel free to use as many code cells as needed.


### Q6. Run the predictions here.
### Feel free to use as many code cells as needed.

### Q7. Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.

### Q8. Use the model's softmax probabilities to visualize the certainty of its predictions