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
num_channels = 1 # grey scale

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

plt.show()


### Step 1. Preprocess the data here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
# reformat to grayscale
X_train = X_train.reshape((-1, image_shape[0], image_shape[0], num_channels)).astype(np.float32)
X_test = X_test.reshape((-1, image_shape[0], image_shape[0], num_channels)).astype(np.float32)
# create validation set
X_valid, X_test = np.split(X_test, 2)
y_valid, y_test = np.split(y_test, 2)
# normalize
X_train = tf.nn.l2_normalize(X_train, [0, 0, 0])
X_test = tf.nn.l2_normalize(X_test, [0, 0, 0])
X_valid = tf.nn.l2_normalize(X_valid, [0, 0, 0])


### Q1. Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

### Q2. Define your architecture here.
### Feel free to use as many code cells as needed.


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
batch_size = 128
filter_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_shape[0], image_shape[1], num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
    tf_valid_dataset = tf.constant(X_valid)
    tf_test_dataset = tf.constant(X_test)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    #tf.nn.max_pool()
    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, filter=layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, filter=layer2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # dropout layer
    keep_prob = tf.Variable(0.5)
    hidden_layer_drop = tf.nn.dropout(model(tf_train_dataset), keep_prob)

    # Training computation.
    logits = hidden_layer_drop
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # introducing variable learning rate
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.1, global_step, 150, 0.9)
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

### Q3. Train your model here.
### Feel free to use as many code cells as needed.
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

### Step 2. Load the images and plot them here.
### Feel free to use as many code cells as needed.


### Q6. Run the predictions here.
### Feel free to use as many code cells as needed.

### Q7. Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.

### Q8. Use the model's softmax probabilities to visualize the certainty of its predictions