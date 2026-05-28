import numpy as np


def accuracy(y_pred, y_true):

    return np.mean(y_pred == y_true)


def confusion_matrix(y_pred, y_true, num_classes=10):

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(y_pred, y_true):
        cm[t, p] += 1
    return cm