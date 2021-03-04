import tensorflow as tf
import numpy as np
import codecs
from keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_wsd_train_x():
    wsd_train_x = codecs.open('40333_train_data', mode = 'r', encoding= 'utf-8')
    line = wsd_train_x.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[3:]
        list1.append(b)
        line = wsd_train_x.readline()
    return np.array(list1)
    wsd_train_x.close()


def load_wsd_test_x():
    wsd_test_x = codecs.open('40333_test_data', mode = 'r', encoding= 'utf-8')
    line = wsd_test_x.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[3:]
        list1.append(b)
        line = wsd_test_x.readline()
    return np.array(list1)
    wsd_test_x.close()


def load_wsd_train_y():
    wsd_train_y = codecs.open('40333_train_target', mode = 'r', encoding = 'utf-8')
    line = wsd_train_y.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[1:2]
        list1.append(b)
        line = wsd_train_y.readline()
    return (np.array(list1)).reshape(50,)
    wsd_train_y.close()



def load_wsd_test_y():
    wsd_test_y = codecs.open('40333_test_target', mode = 'r', encoding = 'utf-8')
    line = wsd_test_y.readline()
    list1 = []
    while line:
        a = line.split()
        b = a[1:2]
        list1.append(b)
        line = wsd_test_y.readline()
    return (np.array(list1)).reshape(50,)
    wsd_test_y.close()


b = np.zeros(50)

wsd_train_x = load_wsd_train_x()
wsd_test_x = load_wsd_test_x()

wsd_train_y = load_wsd_train_y()
wsd_train_y = to_categorical(wsd_train_y)
#wsd_train_y = np.c_[wsd_train_y, b]

wsd_test_y = load_wsd_test_y()
wsd_test_y = to_categorical(wsd_test_y)
#wsd_test_y = np.c_[wsd_test_y, b]

max_epoch = 100
train_size = wsd_train_x.shape[0]
batch_size = 10
n_batch = train_size // batch_size


layer_num = 2
gogi_num = 3

if layer_num == 3:

    x = tf.placeholder(tf.float32, [None, 768])
    y = tf.placeholder(tf.float32, [None, gogi_num])

    W1 = tf.Variable(tf.zeros([768, 50]))
    b1 = tf.Variable(tf.zeros([50]))
    L1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.zeros([50, gogi_num]))
    b2 = tf.Variable(tf.zeros[gogi_num])

    predict = tf.nn.softmax(tf.matmul(L1, W2) + b2)


if layer_num == 2:

    x = tf.placeholder(tf.float32, [None, 768])
    y = tf.placeholder(tf.float32, [None, gogi_num])

    W = tf.Variable(tf.zeros([768, gogi_num]))
    b = tf.Variable(tf.zeros([gogi_num]))

    predict = tf.nn.softmax(tf.matmul(x, W) + b)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=predict))
train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()

correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(init)

    max_acc = 0.

    for epoch in range(max_epoch):
        batch_mask = np.random.choice(train_size, batch_size)
        for batch in range(n_batch):

            x_batch = wsd_train_x[batch_mask]
            t_batch = wsd_train_y[batch_mask]

        sess.run(train_step, feed_dict={x: x_batch, y: t_batch})
        acc = sess.run(accuracy, feed_dict={x:wsd_test_x, y:wsd_test_y})

        if max_acc < acc.item():
            max_acc = acc.item()
            saver.save(sess, "model/40333_wsd_model_which_retain_with_best_epoch.ckpt")



































