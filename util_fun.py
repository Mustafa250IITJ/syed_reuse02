
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

def split_data(X, y, test_size, dev_size):
    train_size = 1 - test_size - dev_size
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
##    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_size / (dev_size + train_size), shuffle=False)
##    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_size/(1-test_size), random_state=1)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def tune_hparams(X_train, y_train, X_dev, y_dev, param_comb):
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
    
    return best_hparams, best_model, best_accuracy

def train_classifier(X_train, y_train, hyperparameters):
    clf = svm.SVC(**hyperparameters)
    clf.fit(X_train, y_train)
    return clf
