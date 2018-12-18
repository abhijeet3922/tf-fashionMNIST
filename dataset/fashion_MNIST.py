import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Dataset(object):

    def __init__(self, data_download_path="", validation_flag=False, verbose=False):
        self.data_download_path = data_download_path
        self.validation_flag=validation_flag
        self.verbose = verbose
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self):

        if not os.path.exists(self.data_download_path):
            raise ValueError('No "%s" directory found, keep data in the given directory path' %self.data_download_path)

        if not os.path.exists(os.path.join(self.data_download_path,"train-images-idx3-ubyte.gz")):
            raise ValueError('No train images data found, Kindly download the data-set.')

        if not os.path.exists(os.path.join(self.data_download_path,"train-labels-idx1-ubyte.gz")):
            raise ValueError('No train labels data found, Kindly download the data-set.')

        if not os.path.exists(os.path.join(self.data_download_path,"t10k-images-idx3-ubyte.gz")):
            raise ValueError('No test images data found, Kindly download the data-set.')

        if not os.path.exists(os.path.join(self.data_download_path,"t10k-labels-idx1-ubyte.gz")):
            raise ValueError('No test labels data found, Kindly download the data-set.')

        train_images_path = os.path.join(self.data_download_path, "train-images-idx3-ubyte.gz")
        train_label_path = os.path.join(self.data_download_path, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(self.data_download_path, "t10k-images-idx3-ubyte.gz")
        test_label_path = os.path.join(self.data_download_path, "t10k-labels-idx1-ubyte.gz")

        with gzip.open(train_label_path) as train_labelpath:
            y_train = np.frombuffer(train_labelpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(train_images_path) as train_imgpath:
            X_train = np.frombuffer(train_imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_train),784)

        with gzip.open(test_label_path) as test_labelpath:
            y_test = np.frombuffer(test_labelpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(test_images_path) as test_imgpath:
            X_test = np.frombuffer(test_imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_test),784)

        if self.validation_flag:
            if self.verbose:
                print("Dataset split is Train : 54k, Val: 6k, Test: 10k")
            X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      stratify = y_train,
                                                      test_size = 0.1,
                                                      random_state = 42)
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test
            return X_train, X_val, X_test, y_train, y_val, y_test

        if self.verbose:
            print("Dataset split is Train : 60k, Val: 10k")
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test

    def show_samples_in_grid(self, w=0, h=0):
        k = w*h
        for i in range(w):
            for j in range(h):
                plt.subplot2grid((w,h),(i,j))
                plt.imshow(self.X_train[k].reshape(28,28), cmap='Greys')
                plt.axis('off')
                k  = k + 1
        plt.show()

    def create_label_dict(self):
        label_dict = {
         0: 'T-shirt/top',
         1: 'Trouser',
         2: 'Pullover',
         3: 'Dress',
         4: 'Coat',
         5: 'Sandal',
         6: 'Shirt',
         7: 'Sneaker',
         8: 'Bag',
         9: 'Ankle boot'
        }

        return label_dict
