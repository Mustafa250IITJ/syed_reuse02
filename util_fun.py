# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_and_visualize_digits():
    digits = datasets.load_digits()
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    return digits


def preprocess_data(digits):
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data


##def train_and_predict(data, target):
##    # Create a classifier: a support vector classifier
##    clf = svm.SVC(gamma=0.001)
##    # Split data into 50% train and 50% test subsets
##    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=False)
##    # Learn the digits on the train subset
##    clf.fit(X_train, y_train)
##    # Predict the value of the digit on the test subset
##    predicted = clf.predict(X_test)
##    # builds a text report showing the main classification metrics.
##    print(
##        f"Classification report for classifier {clf}:\n"
##        f"{metrics.classification_report(y_test, predicted)}\n"
##    )
##    return X_test, predicted, y_test


def train_classifier(data, target):
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=False)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test

def predict_and_generate_report(classifier, data, target):
    # Predict the value of the digit on the test subset
    predicted = classifier.predict(data)
    # Generate a text report showing the main classification metrics
    report = metrics.classification_report(target, predicted)
    return predicted, report

    
def visualize_predictions(X_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    

def evaluate_classifier(y_test, predicted):
    # plot a :ref:`confusion matrix <confusion_matrix>` of the true digit values and the predicted digit values.
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

##    classification_rep = metrics.classification_report(y_test, predicted)
##    print(f"Classification report:\n{classification_rep}\n")
##
##    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
##    disp = metrics.ConfusionMatrixDisplay(confusion_matrix)
##    disp.plot(cmap=plt.cm.Blues)
##    plt.title("Confusion Matrix")
##    plt.show()
