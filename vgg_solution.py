# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import sys
import random
import matplotlib.pyplot as plt
import argparse
from pandas.core.frame import DataFrame

class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 300
weight_decay = 0.08
dropout_rate = 0.5
momentum_rate = 0.9
model_save_path = './vargroupmodel/modl_ckpt'


## 对下载数据 如果没有数据 就去网站自动下载
def download_data():
    dirname = 'Cifar_10'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = './Cifar_10/cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet already exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


# 读取序列化数据 返回类型字典类型数据
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 将数据导入
def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


# 对 label进行处理
def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])  # 将图片转置 跟卷积层一致
    return data, labels


def prepare_data():
    print("======Loading data======")
    download_data()
    data_dir = './Cifar_10/cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')
    print(meta)
    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))  # 打乱数组
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input, train_flag):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


# 随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


# 对数据随机左右翻转
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


## Z-score 标准化
def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def learning_rate_schedule(epoch_num):
    if epoch_num < 20:
        return 0.1
    elif epoch_num < 90:
        return 0.01
    elif epoch_num < 120:
        return 0.001
    elif epoch_num < 140:
        return 0.0001
    elif epoch_num < 190:
        return 0.00005
    else:
        return 0.000001


def run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob, train_flag):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index + add]
        batch_y = test_y[pre_index:pre_index + add]
        pre_index = pre_index + add
        loss_, acc_ = sess.run([cross_entropy, accuracy],
                               feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
        # loss_, acc_ = sess.run([cross_entropy, accuracy],
        #                        feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        loss += loss_ / 10.0
        acc += acc_ / 10.0

    # summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
    #                             tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss


# 对原始的张量进行保存
def storevariable(sess, variables):
    store = []
    for i in range(len(variables)):
        store += [sess.run(variables[i])]
    return store


# 显示结果
def picshow1(x_p, acc_p, url):
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for i in range(len(x_p)):
        la = "conv_" + str(i + 1)
        plt.plot(x_p[i], acc_p[i], label=la)
    plt.xlabel('%剪枝率')
    plt.ylabel('准确率')
    plt.legend()
    plt.savefig(url)
    plt.show()


# 显示结果
def picshow(index, acc_plt, loss_plt, url):
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    color = ["#FF4500", "#FFFF00"]
    plt.subplot(211)
    plt.plot(index, acc_plt, label="accuracy", color=color[0])
    plt.xlabel('times')
    plt.ylabel('acc')
    plt.legend()
    plt.subplot(212)
    plt.plot(index, loss_plt, label="loss_value", color=color[1])
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(url)
    plt.show()


def L2(weight):
    a = tf.multiply(weight, weight)
    b = 0.5 * tf.reduce_sum(a)
    return b


def groupLasso(weight, w_coff):
    split_conv1 = tf.split(weight, num_or_size_splits=weight.shape[3].value, axis=3)
    list = []
    for i in range(len(split_conv1)):
        list.append(L2(split_conv1[i]))
    s = list[0]
    for i in range(len(list) - 1):
        s = s + list[i + 1]
    s = s * w_coff
    return s


# 对卷积核剪切 var是数组，index也是数组
def cut_filter(var, bias, p):
    v = np.reshape(var, (-1, var.shape[-1]))
    v = np.abs(v)
    v = np.sum(v, axis=0)
    ind = np.argsort(v)
    t = int(len(ind) * p)
    ind = ind[0:t]
    for i in range(var.shape[-1]):
        if i in ind:
            var[:, :, :, i] = 0
            bias[i] = 0
    var = var.astype(np.float32)
    bias = bias.astype(np.float32)
    return var, bias


def wpenatly(weight):
    mean, var = tf.nn.moments(weight, axes=[0, 1, 2])
    value, index = tf.nn.top_k(var, int(weight.shape[3].value * 0.7))
    one = tf.ones_like(weight)
    # one = tf.ones([weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]])

    l = tf.greater(var, value[-1])
    l = tf.cast(l, dtype=float)
    list = []
    for i in range(weight.shape[3]):
        temp = one[:, :, :, i] * l[i]
        list.append(temp)
    st = tf.stack(list, axis=3)
    label = tf.where(st == 0, st, weight)
    weight = tf.assign(weight, label)
    # weight=st*weight
    return weight


# vgg16 网络结构
def vgg16(x, keep_prob, regularizer, train_flag):
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    tensors = []

    # build_network
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    # W_conv1_1 = wpenatly(W_conv1_1)
    tensors += [W_conv1_1, b_conv1_1]
    output = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1) + b_conv1_1, train_flag))
    # output = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    # W_conv1_2 = wpenatly(W_conv1_2)
    tensors += [W_conv1_2, b_conv1_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv1_2) + b_conv1_2)
    output = max_pool(output, 2, 2, "pool1")
    # out :16

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    # W_conv2_1 = wpenatly(W_conv2_1)
    tensors += [W_conv2_1, b_conv2_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv2_1) + b_conv2_1)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    # W_conv2_2 = wpenatly(W_conv2_2)
    tensors += [W_conv2_2, b_conv2_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv2_2) + b_conv2_2)
    output = max_pool(output, 2, 2, "pool2")
    # out :8

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    # W_conv3_1 = wpenatly(W_conv3_1)
    tensors += [W_conv3_1, b_conv3_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1) + b_conv3_1, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv3_1) + b_conv3_1)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    # W_conv3_2 = wpenatly(W_conv3_2)
    tensors += [W_conv3_2, b_conv3_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv3_2) + b_conv3_2)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    # W_conv3_3 = wpenatly(W_conv3_3)
    tensors += [W_conv3_3, b_conv3_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3) + b_conv3_3, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv3_3) + b_conv3_3)
    output = max_pool(output, 2, 2, "pool3")
    # out :4

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    # W_conv4_1 = wpenatly(W_conv4_1)
    tensors += [W_conv4_1, b_conv4_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv4_1) + b_conv4_1)

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    # W_conv4_2 = wpenatly(W_conv4_2)
    tensors += [W_conv4_2, b_conv4_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv4_2) + b_conv4_2)

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    # W_conv4_3 = wpenatly(W_conv4_3)
    tensors += [W_conv4_3, b_conv4_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv4_3) + b_conv4_3)
    output = max_pool(output, 2, 2)
    # out :2

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    # W_conv5_1 = wpenatly(W_conv5_1)
    tensors += [W_conv5_1, b_conv5_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv5_1) + b_conv5_1)

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    # W_conv5_2 = wpenatly(W_conv5_2)
    tensors += [W_conv5_2, b_conv5_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv5_2) + b_conv5_2)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    # W_conv5_3 = wpenatly(W_conv5_3)
    tensors += [W_conv5_3, b_conv5_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3, train_flag))
    # output = tf.nn.relu(conv2d(output, W_conv5_3) + b_conv5_3)
    # output = max_pool(output, 2, 2)

    # output = tf.contrib.layers.flatten(output)
    flatten_output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])

    output = tf.nn.relu(batch_norm(tf.matmul(flatten_output, W_fc1) + b_fc1, train_flag))
    # output = tf.nn.relu(tf.matmul(flatten_output, W_fc1) + b_fc1)
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])

    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2, train_flag))
    # output = tf.nn.relu(tf.matmul(output, W_fc2) + b_fc2)
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.
                            he_normal())
    b_fc3 = bias_variable([10])

    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3, train_flag))
    if regularizer != None:
        for i in range(13):
            tf.add_to_collection('losses', groupLasso(tensors[2 * i], regularizer))
    return output, tensors, train_x, train_y, test_x, test_y


# 训练
def expriment():
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    output, tensors, train_x, train_y, test_x, test_y = vgg16(x, keep_prob, weight_decay, train_flag)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(
        cross_entropy + l2)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_accuracy = []
        train_accuracy = []
        end = False
        max_evl = 0
        for ep in range(1, total_epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()
            print("\n epoch %d/%d:" % (ep, total_epoch))
            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                    learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    # loss_, acc_= sess.run([cross_entropy, accuracy],feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0,
                    #                                            train_flag: True})
                    val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob,
                                                    train_flag)
                    test_accuracy.append(val_acc)
                    train_accuracy.append(train_acc)
                    if val_acc > max_evl:
                        max_evl = val_acc
                        saver.save(sess, model_save_path, ep)
                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (
                              it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                    if val_acc > 0.9100:
                        end = True
            if end == True:
                break

        np.save("./gltest_acc.npy", test_accuracy)
        np.save("./gltrain_acc.npy", train_accuracy)
        t = sess.run(tensors)
        np.savez('./groupconv.npz', conv1=t[0], conv2=t[2], conv3=t[4], conv4=t[6], conv5=t[8], conv6=t[10],
                 conv7=t[12],
                 conv8=t[14], conv9=t[16], conv10=t[18], conv11=t[20], conv12=t[22], conv13=t[24])


def storevariable(sess, variables):
    store = []
    for i in range(len(variables)):
        store += [sess.run(variables[i])]
    return store


## 测试
def evel():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        y_ = tf.placeholder(tf.float32, [None, class_num])
        keep_prob = tf.placeholder(tf.float32)
        train_flag = tf.placeholder(tf.bool)
        output, tensors, train_x, train_y, test_x, test_y = vgg16(x, keep_prob, weight_decay, train_flag)
        ckpt = tf.train.get_checkpoint_state('./penaltygroupmodel')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            store_wb = storevariable(sess, tensors)
            val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob,
                                            train_flag)
            print(val_acc, val_loss)
            print("----------------------")
            for i in range(13):
                mean, var = tf.nn.moments(tensors[2 * i], axes=[0, 1, 2])
                value, index = tf.nn.top_k(var, int(tensors[2 * i].shape[3].value * 0.5))
                one = tf.ones_like(tensors[2 * i])
                # one = tf.ones([weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]])

                l = tf.greater(var, value[-1])
                l = tf.cast(l, dtype=float)
                list = []
                for j in range(tensors[2 * i].shape[3]):
                    temp = one[:, :, :, j] * l[j]
                    list.append(temp)
                st = tf.stack(list, axis=3)
                label = tf.where(tf.equal(st, 0), st, tensors[2 * i])
                # print(sess.run(tf.count_nonzero(st)))
                # print(sess.run(tf.count_nonzero(label)))
                sess.run(tf.assign(tensors[2 * i], label))
                print(tensors[2 * i].dtype)
            #                 print(tensors[2*i].shape[0].value*tensors[2*i].shape[1].value*tensors[2*i].shape[2].value*tensors[2*i].shape[3].value)
            #                 print(sess.run(tf.count_nonzero(tensors[2*i])))
            val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob,
                                            train_flag)
            print(val_acc, val_loss)
            ##剪枝测试
            store_wb_1 = storevariable(sess, tensors)
            wb = store_wb_1.copy()
            x_p = []
            acc_p = []
            for i in range(13):
                x_p.append([])
                acc_p.append([])
                for p in range(0, 10, 1):
                    if p == 0:
                        val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob,
                                                        train_flag)
                        x_p[i].append(p * 10)
                        acc_p[i].append(val_acc)
                        print("the pruning %f after pruning test accuracy %g" % (p / 10, val_acc))
                        original_w = sess.run(tensors[2 * i])
                        original_b = sess.run(tensors[2 * i + 1])
                    else:
                        var, b = cut_filter(wb[2 * i], wb[2 * i + 1], p / 10)
                        #                         print(var.shape)
                        a = sess.run(tf.assign(tensors[2 * i], var))
                        aa = sess.run(tf.assign(tensors[2 * i + 1], b))
                        val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob,
                                                        train_flag)
                        x_p[i].append(p * 10)
                        acc_p[i].append(val_acc)
                        print("the pruning %f after pruning test accuracy %g" % (p / 10, val_acc))
                        if p == 9:
                            sess.run(tf.assign(tensors[2 * i], original_w))
                            sess.run(tf.assign(tensors[2 * i + 1], original_b))
                            val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_,
                                                            keep_prob,
                                                            train_flag)
                            print(val_acc, val_loss)
            np.savez("./acc_W_penatly.npz", conv1=acc_p[0], conv2=acc_p[1], conv3=acc_p[2], conv4=acc_p[3],
                     conv5=acc_p[4], conv6=acc_p[5],
                     conv7=acc_p[6],
                     conv8=acc_p[7], conv9=acc_p[8], conv10=acc_p[9], conv11=acc_p[10], conv12=acc_p[11],
                     conv13=acc_p[12])
            picshow1(x_p, acc_p, "./prueglasfilter.jpg")


def finetune():
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    output, tensors, train_x, train_y, test_x, test_y = vgg16(x, keep_prob, weight_decay, train_flag)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(
        cross_entropy + l2)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./grouplassomodel')
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
        test_accuracy = []
        train_accuracy = []
        end = False
        max_evl = 0
        for ep in range(1, total_epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()
            print("\n epoch %d/%d:" % (ep, total_epoch))
            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                    learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size
                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    # loss_, acc_= sess.run([cross_entropy, accuracy],feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0,
                    #                                            train_flag: True})
                    val_acc, val_loss = run_testing(sess, test_x, test_y, cross_entropy, accuracy, x, y_, keep_prob,
                                                    train_flag)
                    test_accuracy.append(val_acc)
                    train_accuracy.append(train_acc)
                    if val_acc > max_evl:
                        max_evl = val_acc
                        saver.save(sess, model_save_path, ep)
                        t = sess.run(tensors)
                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (
                              it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
        #                     if val_acc > 0.9100:
        #                         end = True
        #             if end == True:
        #                 break

        np.save("./finetunetest_acc.npy", test_accuracy)
        np.save("./finetunetrain_acc.npy", train_accuracy)

        np.savez('./finetuneconv.npz', conv1=t[0], conv2=t[2], conv3=t[4], conv4=t[6], conv5=t[8], conv6=t[10],
                 conv7=t[12],
                 conv8=t[14], conv9=t[16], conv10=t[18], conv11=t[20], conv12=t[22], conv13=t[24])


# if __name__ == '__main__':
evel()