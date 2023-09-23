"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from util_fun import split_data, tune_hparams, train_classifier
from itertools import product

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# Load the digits dataset
digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

C_ranges = [0.1, 1, 2, 5, 10]
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]



# Create a list of dictionaries for hyperparameter combinations
param_comb = [{'C': C, 'gamma': gamma} for C in C_ranges for gamma in gamma_ranges]

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]
param_combinations = product(test_sizes, dev_sizes)



for test_size, dev_size in param_combinations:
    # split the data into train, dev, and test sets
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(data, digits.target, test_size, dev_size)

    # hyperparameter tuning on the train and dev sets
    best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, param_comb)

    # train the classifier
    clf = train_classifier(X_train, y_train, best_hparams)

    # predictions on train, dev, and test data
    y_train_pred = clf.predict(X_train)
    y_dev_pred = clf.predict(X_dev)
    y_test_pred = clf.predict(X_test)

    # finding the accuracy on train, dev, and test data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # printing the output as per assignment3 
    #print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size:0.1f}\n"
          #f"train_acc={train_accuracy:.2f} dev_acc={dev_accuracy:.2f} test_acc={test_accuracy:.2f}")
    #print(f"Best Hyperparameters: {best_hparams}\n")

total_samples = len(digits.images)
print(f"Total number of samples in the dataset: {total_samples}")


image_height, image_width = digits.images.shape[1], digits.images.shape[2]
print(f"Size of the images in the dataset (height x width): {image_height} x {image_width}")



