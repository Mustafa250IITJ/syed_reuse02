import numpy as np
from skimage.transform import resize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from util_fun import split_data,tune_hparams,train_classifier


def resize_and_evaluate(image_size, X, y, test_size, dev_size, param_comb):
    resized_images = np.array([resize(image, (image_size, image_size)) for image in X])
    n_samples = len(resized_images)
    data = resized_images.reshape((n_samples, -1))
    
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(data, y, test_size, dev_size)
    
    best_accuracy = 0
    best_model = None
    best_hparams = None
    
    for param_combination in param_comb:
        clf = svm.SVC(**param_combination)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_dev)
        accuracy = accuracy_score(y_dev, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hparams = param_combination
            best_model = clf
    
    clf = train_classifier(X_train, y_train, best_hparams)
    
    y_train_pred = clf.predict(X_train)
    y_dev_pred = clf.predict(X_dev)
    y_test_pred = clf.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Image size: {image_size}x{image_size} train_size: {1 - test_size - dev_size:.1f} dev_size: {dev_size:.1f} test_size: {test_size:.1f}")
    print(f"Train accuracy: {train_accuracy:.2f} Dev accuracy: {dev_accuracy:.2f} Test accuracy: {test_accuracy:.2f}")
    print(f"Best Hyperparameters: {best_hparams}\n")

# Load the digits dataset
digits = datasets.load_digits()

image_sizes = [4, 6, 8]
param_comb = [{'C': 0.1, 'gamma': 0.001}, {'C': 1, 'gamma': 0.01}, {'C': 5, 'gamma': 0.1}]


for size in image_sizes:
    resize_and_evaluate(image_size=size, X=digits.images, y=digits.target, test_size=0.2, dev_size=0.1, param_comb=param_comb)
