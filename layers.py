import tensorflow as tf
import numpy as np

log = lambda x: tf.log(1e-20 + x)
sgm = tf.sigmoid
relu = tf.nn.relu
dense = tf.layers.dense
flatten = tf.contrib.layers.flatten

def conv(x, filters, kernel_size=3, strides=1, **kwargs):
    return tf.layers.conv2d(x, filters, kernel_size, strides,
            data_format='channels_first', **kwargs)

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])

def get_count(nlist):
    avg = sum(nlist)/10.
    count = [np.sqrt(N_t/avg) for N_t in nlist]
    return tf.convert_to_tensor(count, dtype=tf.float32)

def ovr_cross_entropy(logits, labels):
    y = labels
    yhat = sgm(logits)
    cent = -y * log(yhat) - (1-y) * log(1-yhat)
    return cent

def l2_decay(decay, var_list=None):
    var_list = tf.trainable_variables() if var_list is None else var_list
    return decay*tf.add_n([tf.nn.l2_loss(var) for var in var_list])

def l1_decay(decay, var_list=None):
    return decay*tf.reduce_sum(tf.add_n([tf.abs(var) for var in var_list]))

def accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
