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

## 参数设置
argparser = argparse.ArgumentParser()
argparser.add_argument("-1", "--train", action="store_true",
    help="train vgg16 model with 164 iterations")
argparser.add_argument("-2", "--filterprune", action="store_true",
    help="prune model by filter num")
argparser.add_argument("-3","--singlekernel",action="store_true",
                       help="prune model by single kernel")
argparser.add_argument("-4","--layerprune",action="store_true",
                      help="prune model by layer")
argparser.add_argument("-5", "--SpareFilterprune", action="store_true",
                      help="prune model by spare filter")
argparser.add_argument("-6", "--test", action="store_true",
                      help="test the number")
args = argparser.parse_args()

class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 164
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_16_logs'
model_save_path = './model/modl_ckpt'

# 用来计算权重和偏置的均值和方差
def variable_summaries(var):
    # 统计参数的均值,并记录
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
    # 计算参数的标准差
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
    # 统计参数的最大最小值
    tf.summary.scalar("max", tf.reduce_max(var))
    tf.summary.scalar("min", tf.reduce_min(var))
    # 用直方图统计参数的分布
    tf.summary.histogram("histogram", var)

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
            percent = min(int(count*block_size*100/total_size),100)
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
    indices = np.random.permutation(len(train_data)) #打乱数组
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


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


# 随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])


    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
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
def data_preprocessing(x_train,x_test):


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
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001


def run_testing(sess, ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index = pre_index + add
        loss_, acc_  = sess.run([cross_entropy, accuracy],
                                feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
        # loss_, acc_ = sess.run([cross_entropy, accuracy],
        #                        feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
        loss += loss_ / 10.0
        acc += acc_ / 10.0

    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss), 
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss, summary



# 对原始的张量进行保存
def storevariable(sess,variables):
    store = []
    for i in range(len(variables)):
        store += [sess.run(variables[i])]
    return store

# 对卷积核剪切 var是数组，index也是数组
def cut_filter(var,bias,index):
    var_shape = var.shape
    bias_shape = bias.shape
    s_len = len(index)
    w = np.zeros((var_shape[0], var_shape[1], var_shape[2], var_shape[3]))
    b = np.zeros((bias_shape[0]))
    j = 0
    for i in range(var_shape[3]):
        if i in index:
            w[:, :, :, j] = var[:, :, :, i]
            b[j] = bias[i]
        j = j+1
    tensor_w = tf.convert_to_tensor(w,dtype=tf.float32)
    tensor_b = tf.convert_to_tensor(b,dtype=tf.float32)
    return tensor_w,tensor_b

# 对featuremap 修减
def cut_featuremap(var,index):
    var_shape = var.shape
    s_len = len(index)
    w = np.zeros((var_shape[0], var_shape[1], var_shape[2], var_shape[3]))
    j = 0
    for i in range(var_shape[2]):
        if i in index:
            w[:, :, j, :] = var[:, :, i, :]
        j = j + 1
    tensor_w = tf.convert_to_tensor(w,dtype=tf.float32)
    return tensor_w

# 对数据进行操作 返回剪掉核之后保留的序列号
def alter_weight(var,p):
    l = var.shape.as_list()
    var_shape = tf.reshape(var,[-1,l[-1]])
    abs_shape = tf.abs(var_shape)
    w = tf.reduce_sum(abs_shape,0)
    max_val = tf.reduce_max(w)
    value, index = tf.nn.top_k(w, l[-1])
    pp = round((1-p) * l[-1])
    s_w = tf.slice(index, [0], [pp])
    return s_w

# 剪枝后 微调
def fine_tine(sess):
    pre_index = 0
    train_acc = 0.0
    train_loss = 0.0
    for i in range(200):
        batch_x = train_x[pre_index:pre_index + batch_size]
        batch_y = train_y[pre_index:pre_index + batch_size]

        batch_x = data_augmentation(batch_x)
        _ = sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                            learning_rate: 0.001, train_flag: True})
        batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})
        train_acc += batch_acc
        pre_index += batch_size
    print("fine tine accuracy is %g"%(train_acc/200))

# 返回保存的数据
def filterpruning(tensors,store_wb):
    x_p = []
    acc_p = []
    for i in range(12):
        x_p.append([])
        acc_p.append([])
        for p in range(0, 10, 1):
            sava_index = alter_weight(tensors[2*i], p / 10)  # 剪掉后剩余的下标
            array_index = sess.run(sava_index)
            w_after, b_after = cut_filter(store_wb[2*i], store_wb[2*i+1], array_index)
            sess.run(tf.assign(tensors[2*i], w_after))
            sess.run(tf.assign(tensors[2*i+1], b_after))
            # W_conv1_1 = w_after
            # b_conv1_1 = b_after
            fm_after = cut_featuremap(store_wb[2*i+2], array_index)
            sess.run(tf.assign(tensors[2*i+2], fm_after))
            # W_conv1_2 = fm_after
            ## fine-tine
            # fine_tine(sess)
            accuracy_score_p, loss, _ = run_testing(sess, p)
            x_p[i].append(p * 10)
            acc_p[i].append(accuracy_score_p)
            print("the pruning %f after pruning test accuracy %g" % (p / 10, accuracy_score_p))
        sess.run(tf.assign(tensors[2*i], w_after))
        sess.run(tf.assign(tensors[2*i+1], b_after))
        sess.run(tf.assign(tensors[2*i+2], fm_after))
    return x_p,acc_p
## 对全局数据排序，返回数据剪枝索引
def alter_all(conv,p):
    re_conv = np.reshape(conv,(-1,conv.shape[-2],conv.shape[-1]))
    re_conv = np.abs(re_conv)
    re_conv = np.sum(re_conv,axis=0)
    line_w = []  ## 将数据变成一维
    for i in range(re_conv.shape[-1]):
        for j in range(re_conv.shape[-2]):
            line_w.append(re_conv[j][i])
    index = np.argsort(line_w)
    pp =round(p*len(line_w))
    if(pp == 0):
        store_index = []
    else:
        store_index = index[0:pp-1]  ##保存p后的数据
    return store_index
## 对全局剪枝
def all_pruning(weight,index):
    if len(index) == 0:
        return weight
    else:
        weight_p = weight.copy()
        for i in range(len(index)):
            a =int(index[i] % weight.shape[-2])
            b = int(index[i] / weight.shape[-1])
            weight_p[:,:,a,b]=0
        return weight_p
# 全局剪枝
def allpruning(tensors,conv):
    x_p = []
    acc_p = []
    for i in range(13):
        x_p.append([])
        acc_p.append([])
        for p in range(0, 10, 1):
            index = alter_all(conv[2*i],p/10)
            weight = all_pruning(conv[2*i],index)
            sess.run(tf.assign(tensors[2*i],weight))
            accuracy_score_p, loss, _ = run_testing(sess, p)
            print("the %d pruning %f after pruning test accuracy %g" % (i+1,p/10, accuracy_score_p))
            x_p[i].append(p*10)
            acc_p[i].append(accuracy_score_p)
        sess.run(tf.assign(tensors[2*i],conv[2*i]))
    return x_p,acc_p

# 对层中剪枝
def layer_prune(conv,p):
    prune_conv = conv.copy()
    if p == 0:
        return conv
    else:
        re_conv = np.reshape(conv, (-1, conv.shape[-2], conv.shape[-1]))
        re_conv = np.abs(re_conv)
        re_conv = np.sum(re_conv, axis=0)
        pp = round(p * conv.shape[-1])  # 返回剪枝的个数
        for i in range(conv.shape[-2]):
            sort_w = sorted(re_conv[i])
            t = sort_w[pp]
            for j in range(conv.shape[-1]):
                if re_conv[i][j] < t:
                    prune_conv[:,:,i,j] = 0
        return prune_conv
# 对层按比例剪枝
def layerpruning(tensors,conv):
    x_p = []
    acc_p = []
    for i in range(12):
        x_p.append([])
        acc_p.append([])
        for p in range(0, 10, 1):
            weight = layer_prune(conv[2*i],p/10)
            sess.run(tf.assign(tensors[2*i],weight))
            accuracy_score_p, loss, _ = run_testing(sess, p)
            print("the %d pruning %f after pruning test accuracy %g" % (i+1,p/10, accuracy_score_p))
            x_p[i].append(p*10)
            acc_p[i].append(accuracy_score_p)
        sess.run(tf.assign(tensors[2*i],conv[2*i]))
    return x_p,acc_p

# 对过滤器按比例剪枝
def sparefilter_prune(conv,p):
    prune_conv = conv.copy()
    if p == 0:
        return prune_conv
    else:
        re_conv = np.reshape(conv, (-1, conv.shape[-2], conv.shape[-1]))
        re_conv = np.abs(re_conv)
        re_conv = np.sum(re_conv, axis=0)
        re_conv = re_conv.T
        pp = round(p * conv.shape[-2])  # 返回剪枝的个数
        for i in range(conv.shape[-1]):
            sort_w = sorted(re_conv[i])
            if pp == 3:
                pp=2
            t = sort_w[pp]
            for j in range(conv.shape[-2]):
                if re_conv[i][j] < t:
                    prune_conv[:,:,j,i] = 0
        return prune_conv
def sparefilterpruning(tensors,conv):
    x_p = []
    acc_p = []
    for i in range(12):
        x_p.append([])
        acc_p.append([])
        for p in range(0, 10, 1):
            weight = sparefilter_prune(conv[2*i],p/10)
            sess.run(tf.assign(tensors[2*i],weight))
            accuracy_score_p, loss, _ = run_testing(sess, p)
            print("the %d pruning %f after pruning test accuracy %g" % (i+1,p/10, accuracy_score_p))
            x_p[i].append(p*10)
            acc_p[i].append(accuracy_score_p)
        sess.run(tf.assign(tensors[2*i],conv[2*i]))
    return x_p,acc_p

# 显示结果
def picshow(x_p,acc_p,url):
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    color = ["#FF4500", "#FFFF00", "#FF00FF", "#8B0000", "#030303", "#00F5FF"]
    for i in range(len(x_p)):
        la = "conv_" + str(i + 1)
        if i < 6:
            plt.plot(x_p[i], acc_p[i], label=la, color=color[i])
        else:
            plt.plot(x_p[i], acc_p[i], label=la, color=color[i - 6], linestyle="--")
    plt.xlabel('%剪枝率')
    plt.ylabel('准确率')
    plt.legend()
    plt.savefig(url)
    plt.show()
def Corr_sparefilterpruning(tensors,conv):
    x_p = []
    acc_p = []
    for i in range(12):
        x_p.append([])
        acc_p.append([])
        for p in range(0, 10, 1):
            weight = sparefilter_prune(conv[2*i],p/10)
            weight,lendel = parametercorr(weight)
            print("layer"+str(i)+"  "+str(p)+"剪枝率"+str(lendel))
            sess.run(tf.assign(tensors[2*i],weight))
            accuracy_score_p, loss, _ = run_testing(sess, p)
            print("the %d pruning %f after pruning test accuracy %g" % (i+1,p/10, accuracy_score_p))
            x_p[i].append(p*10)
            acc_p[i].append(accuracy_score_p)
        sess.run(tf.assign(tensors[2*i],conv[2*i]))
    return x_p,acc_p
def parametercorr(conv):
    re_conv = np.reshape(conv, (-1, conv.shape[-2], conv.shape[-1]))
    re_conv = np.abs(re_conv)
    re_conv = np.sum(re_conv, axis=0)
    data = DataFrame(re_conv)
    relation = data.corr()
    length = relation.shape[0]
    final_cols = []
    del_cols = []
    for i in range(length):
        if relation.columns[i] not in del_cols:
            final_cols.append(relation.columns[i])
            for j in range(i + 1, length):
                if (relation.iloc[i, j] > 0.85) and (relation.columns[j] not in del_cols):
                    del_cols.append(relation.columns[j])
    for i in range(len(final_cols)):
        data[final_cols[i]]=0
    nddata =np.array(data)
    data_list = nddata.tolist()
    return data_list,len(del_cols)
## 随机核裁剪
def randomkernel(conv,p):
    prune_conv = conv.copy()
    if p == 0:
        return prune_conv
    else:
        pp = round(p * conv.shape[-1])  # 返回剪枝的个数
        a = range(0,conv.shape[-1],1)
        r = np.random.permutation(a)
        r = r[0:pp]
        for i in range(conv.shape[-1]):
            if i in r:
                prune_conv[:,:,:,i] = 0
##随机裁剪测试
def Randomtest(tensors,conv):
    x_p = []
    acc_p = []
    for i in range(12):
        x_p.append([])
        acc_p.append([])
        for p in range(0, 10, 1):
            weight = randomkernel(conv[2 * i], p / 10)
            sess.run(tf.assign(tensors[2 * i], weight))
            accuracy_score_p, loss, _ = run_testing(sess, p)
            print("the %d pruning %f after pruning test accuracy %g" % (i + 1, p / 10, accuracy_score_p))
            x_p[i].append(p * 10)
            acc_p[i].append(accuracy_score_p)
        sess.run(tf.assign(tensors[2 * i], conv[2 * i]))
    return x_p, acc_p
# vgg16 网络结构
def vgg16(x,keep_prob):
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    # # define placeholder x, y_ , keep_prob, learning_rate
    # x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    # y_ = tf.placeholder(tf.float32, [None, class_num])
    # keep_prob = tf.placeholder(tf.float32)
    # learning_rate = tf.placeholder(tf.float32)
    # train_flag = tf.placeholder(tf.bool)
    tensors = []

    # build_network
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    variable_summaries(W_conv1_1)
    variable_summaries(b_conv1_1)
    tensors += [W_conv1_1, b_conv1_1]
    output = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1) + b_conv1_1))
    # output = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)

    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    variable_summaries(W_conv1_2)
    variable_summaries(b_conv1_2)
    tensors += [W_conv1_2, b_conv1_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2))
    # output = tf.nn.relu(conv2d(output, W_conv1_2) + b_conv1_2)
    output = max_pool(output, 2, 2, "pool1")
    # out :16

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    variable_summaries(W_conv2_1)
    variable_summaries(b_conv2_1)
    tensors += [W_conv2_1, b_conv2_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1))
    # output = tf.nn.relu(conv2d(output, W_conv2_1) + b_conv2_1)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    variable_summaries(W_conv2_2)
    variable_summaries(b_conv2_2)
    tensors += [W_conv2_2, b_conv2_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2))
    # output = tf.nn.relu(conv2d(output, W_conv2_2) + b_conv2_2)
    output = max_pool(output, 2, 2, "pool2")
    # out :8

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    variable_summaries(W_conv3_1)
    variable_summaries(b_conv3_1)
    tensors += [W_conv3_1, b_conv3_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1) + b_conv3_1))
    # output = tf.nn.relu(conv2d(output, W_conv3_1) + b_conv3_1)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    variable_summaries(W_conv3_2)
    variable_summaries(b_conv3_2)
    tensors += [W_conv3_2, b_conv3_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2))
    # output = tf.nn.relu(conv2d(output, W_conv3_2) + b_conv3_2)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    variable_summaries(W_conv3_3)
    variable_summaries(b_conv3_3)
    tensors += [W_conv3_3, b_conv3_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3) + b_conv3_3))
    # output = tf.nn.relu(conv2d(output, W_conv3_3) + b_conv3_3)
    output = max_pool(output, 2, 2, "pool3")
    # out :4

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    variable_summaries(W_conv4_1)
    variable_summaries(b_conv4_1)
    tensors += [W_conv4_1, b_conv4_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1))
    # output = tf.nn.relu(conv2d(output, W_conv4_1) + b_conv4_1)

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    variable_summaries(W_conv4_2)
    variable_summaries(b_conv4_2)
    tensors += [W_conv4_2, b_conv4_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2))
    # output = tf.nn.relu(conv2d(output, W_conv4_2) + b_conv4_2)

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    variable_summaries(W_conv4_3)
    variable_summaries(b_conv4_3)
    tensors += [W_conv4_3, b_conv4_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3))
    # output = tf.nn.relu(conv2d(output, W_conv4_3) + b_conv4_3)
    output = max_pool(output, 2, 2)
    # out :2

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    variable_summaries(W_conv5_1)
    variable_summaries(b_conv5_1)
    tensors += [W_conv5_1, b_conv5_1]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1))
    # output = tf.nn.relu(conv2d(output, W_conv5_1) + b_conv5_1)

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    variable_summaries(W_conv5_2)
    variable_summaries(b_conv5_2)
    tensors += [W_conv5_2, b_conv5_2]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2))
    # output = tf.nn.relu(conv2d(output, W_conv5_2) + b_conv5_2)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    variable_summaries(W_conv5_3)
    variable_summaries(b_conv5_3)
    tensors += [W_conv5_3, b_conv5_3]
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3))
    # output = tf.nn.relu(conv2d(output, W_conv5_3) + b_conv5_3)
    # output = max_pool(output, 2, 2)

    # output = tf.contrib.layers.flatten(output)
    flatten_output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    variable_summaries(W_fc1)
    variable_summaries(b_fc1)
    output = tf.nn.relu(batch_norm(tf.matmul(flatten_output, W_fc1) + b_fc1))
    # output = tf.nn.relu(tf.matmul(flatten_output, W_fc1) + b_fc1)
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])
    variable_summaries(W_fc2)
    variable_summaries(b_fc2)
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
    # output = tf.nn.relu(tf.matmul(output, W_fc2) + b_fc2)
    output = tf.nn.dropout(output,     keep_prob)
 
    W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.
                            he_normal())
    b_fc3 = bias_variable([10])
    variable_summaries(W_fc3)
    variable_summaries(b_fc3)
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))
    return output,tensors,train_x, train_y, test_x, test_y

if __name__ == '__main__':

    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)
    # 返回vgg16 输出，tensors是保存的tensors常量 train_x, train_y, test_x, test_y 数据预处理过的
    output,tensors,train_x, train_y, test_x, test_y = vgg16(x,keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])


    # 里面的系数
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).\
        minimize(cross_entropy + l2 * weight_decay)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 将所有的summary整合
    merged = tf.summary.merge_all()
    # initial an saver to save model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path,sess.graph)

        # epoch = 164 
        # make sure [bath_size * iteration = data_set_number]
        if(args.train == True):
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

                        loss_, acc_, merged_ = sess.run([cross_entropy, accuracy, merged],
                                                        feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0,
                                                                   train_flag: True})
                        train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                          tf.Summary.Value(tag="train_accuracy",
                                                                           simple_value=train_acc)])

                        val_acc, val_loss, test_summary = run_testing(sess, ep)

                        summary_writer.add_summary(merged_, ep)
                        summary_writer.add_summary(train_summary, ep)
                        summary_writer.add_summary(test_summary, ep)
                        summary_writer.flush()
                        saver.save(sess, model_save_path, ep)
                        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                              "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                              % (
                              it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                    else:
                        print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                              % (it, iterations, train_loss / it, train_acc / it))
        if(args.filterprune == True):
            ckpt = tf.train.get_checkpoint_state('./model')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            global_step = ckpt.all_model_checkpoint_paths[-1].split('/')[-1].split('-')[-1]
            # accuracy_score,loss ,_ =run_testing(sess,global_step)
            # print("After %s traing test accuracy is %g,the loss is %g" % (global_step, accuracy_score,loss))
            # ##保留不变的权值
            store_wb = storevariable(sess,tensors)
            x_p,acc_p = filterpruning(tensors,store_wb)
            picshow(x_p,acc_p,"./vgg16filterprune.jpg")
        if(args.singlekernel == True):
            ckpt = tf.train.get_checkpoint_state('./model')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            store_wb = storevariable(sess, tensors) ## 保留不变的值
            x_p,acc_p = allpruning(tensors,store_wb)
            picshow(x_p,acc_p,"./vgg16all_prune_by_single.jpg")
        if(args.layerprune == True):
            ckpt = tf.train.get_checkpoint_state('./model')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            store_wb = storevariable(sess, tensors)  ## 保留不变的tensor值
            x_p, acc_p = layerpruning(tensors,store_wb)
            np.savez("./layer_prune",x_p,acc_p)
            picshow(x_p,acc_p,"./layer_prune.png")
        if (args.SpareFilterprune == True):
            ckpt = tf.train.get_checkpoint_state('./model')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            store_wb = storevariable(sess, tensors)  ## 保留不变的tensor值
            x_p, acc_p = sparefilterpruning(tensors, store_wb)
            np.savez("./spareFilter_prune", x_p, acc_p)
            picshow(x_p, acc_p, "./spareFilter_prune.png")
        if (args.test == True):
            ckpt = tf.train.get_checkpoint_state('./model')
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
            store_wb = storevariable(sess, tensors)  ## 保留不变的tensor值
            x,p=Corr_sparefilterpruning(tensors,store_wb)
            np.savez("./zz", x_p, acc_p)
            picshow(x_p, acc_p, "./zz.png")