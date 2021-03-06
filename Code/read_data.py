import os
import struct
import numpy as np

def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,
                               'train-labels-idx1-ubyte')
    images_path = os.path.join(path,
                               'train-images-idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_train,y_train = load_mnist_train('/Users/cyu/Desktop/MNIST_Dataset_Loader/dataset_train')
np.savetxt('train_img.csv', X_train,
           fmt='%i', delimiter=',')
np.savetxt('train_labels.csv', y_train,
           fmt='%i', delimiter=',')

def load_mnist_test(path, kind='train'):
    labels_path = os.path.join(path,
                               't10k-labels-idx1-ubyte')
    images_path = os.path.join(path,
                               't10k-images-idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_test,y_test = load_mnist_test('/Users/cyu/Desktop/MNIST_Dataset_Loader/dataset_test')
np.savetxt('test_img.csv', X_test,
           fmt='%i', delimiter=',')
np.savetxt('test_labels.csv', y_test,
           fmt='%i', delimiter=',')
