import os
import gzip
import numpy as np

#dir_path = "data/fashion/"

class Dataset(object):

    def __init__(self, data_download_path="", verbose=False):
        self.data_download_path = data_download_path
        self.verbose = verbose

    def load_data(validation_set=False):

        if not os.path.exists(self.data_download_path):
            raise ValueError('No "%s" directory found, keep data in the given directory path' %dir_path)

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

        if validation_set:
            if self.verbose:
                print("Dataset split is Train : 54k, Val: 6k, Test: 10k")
            X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      stratify = y_train,
                                                      test_size = 0.1,
                                                      random_state = 42)
            return X_train, X_val, X_test, y_train, y_val, y_test

        if self.verbose:
            print("Dataset split is Train : 60k, Val: 10k")

        return X_train, X_test, y_train, y_test

    def show_samples_in_grid(w=0, h=0):
        k = w*h
        for i in range(w):
            for j in range(h):
                plt.subplot2grid((w,h),(i,j))
                plt.imshow(X_train[k].reshape(28,28), cmap='Greys')
                plt.axis('off')
                k  = k + 1
        plt.show()

    def create_label_dict():
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