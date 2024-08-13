import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import tensorflow as tf
import tensorflow_datasets as tfds
img_len = 16
t_total = 2
c = 0.238
mean = np.array([-0.97037244, -0.79086132, -0.70268262, -0.91902397, -0.89255285, -0.45840487,
 -0.33903784, -0.80714807, -0.86354743, -0.41002755, -0.34796831, -0.83375346,
 -0.9248745,  -0.643027,   -0.67912024, -0.94334359])


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
n=4

def reshape(image, label):
    image = tf.image.resize(image, [n, n], method='gaussian', antialias=True)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return tf.reshape(image, [-1]), label


ds_train = ds_train.map(reshape, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(reshape, num_parallel_calls=tf.data.AUTOTUNE)
imgs = []
for i, l in ds_train.take(1):
    imgs.append(i.numpy())
imgs = np.array(imgs)
for i in imgs:
    img = i
    def pc(t):
        if t == 0 or t >= t_total: return img[0]
        return img[int(t / t_total * img_len)]
    time = np.linspace(0, t_total, 10000)
    signal = np.array([pc(t) for t in time])
    signal = np.pad(signal, (1, 1), 'constant')
    W = fftfreq(signal.size, d=time[1] - time[0])
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    #cut_f_signal[(abs(W) > 30e6)] = 0
    cut_signal = irfft(cut_f_signal)



    plt.subplot(111)
    plt.plot(time, cut_signal)
    plt.show()