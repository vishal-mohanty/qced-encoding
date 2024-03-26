import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.linear_model import LogisticRegression
trainSize = 4000
testSize = 1000

# Resizes images to 4x4 using gaussian filter and normalises values to the range -1 to 1
def reshape(image, label):
    image = tf.image.resize(image, [4, 4], method='gaussian', antialias=True)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return tf.reshape(image, [-1]), label

# Load datasets from tensorflow, and take a random sample of images
ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)
trainSize_total = ds_train.cardinality().numpy()
testSize_total = ds_test.cardinality().numpy()
ds_train = ds_train.shuffle(trainSize_total)
ds_test = ds_test.shuffle(testSize_total)
ds_train = ds_train.map(reshape)
ds_test = ds_test.map(reshape)

train_X, train_y, test_X, test_y = [], [], [], []
for x, y in ds_train.take(trainSize):
    train_X.append(x.numpy())
    train_y.append(y.numpy())
for x, y in ds_test.take(testSize):
    test_X.append(x.numpy())
    test_y.append(y.numpy())

# Logistic Regression Model
clf = LogisticRegression(max_iter=1000)
clf.fit(train_X, train_y)
score = clf.score(test_X, test_y)
print(f"Accuracy of Logistic Regression: {score*100}%")