# tf-fashionMNIST

Deep Learning with Tensorflow

1. This repo implements each step in building deep learning model from scratch using python & Tensorflow.
2. Most tutorials/blogs/implementations import datasets from APIs like tensorflow/keras etc. We won't, instead we will make our data loader.
3. We will not use any high level APIs like keras or tf.keras etc. We will stick to basic tensorflow

## Fashion MNIST Dataset

* Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples.
* Each example is a 28x28 grayscale image, associated with a label from 10 classes.
* Fashion-MNIST is a direct drop-in replacement for the original MNIST digit dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## Why Fashion MNIST ?

* MNIST is too easy
* MNIST is overused
* MNIST can not represent modern CV tasks (like Batchnorm)

Deep Learning heros like Ian Goodfellow & Francois Chollet have advised serious researchers to stay away from digit recognition MNIST.

I have downloaded the .gz files of train and test data along with labels from https://github.com/zalandoresearch/fashion-mnist#get-the-data

Each training and test example is assigned to one of the following labels:

Label Description

* 0 T-shirt/top
* 1 Trouser
* 2 Pullover
* 3 Dress
* 4 Coat
* 5 Sandal
* 6 Shirt
* 7 Sneaker
* 8 Bag
* 9 Ankle boot

You can see the sample images in Fashion_MNIST_samples.png.

I have written couple of blog-posts illustrating this repository explaining tensorflow and neural network models.
1. https://appliedmachinelearning.blog/2018/12/26/tensorflow-tutorial-from-scratch-building-a-deep-learning-model-on-fashion-mnist-dataset-part-1/
2. https://appliedmachinelearning.blog/2019/01/01/tensorflow-tutorial-from-scratch-building-a-deep-learning-model-on-fashion-mnist-dataset-part-2/


