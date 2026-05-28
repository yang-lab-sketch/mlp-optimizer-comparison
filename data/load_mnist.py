# 备注：Ubuntu 22.04 + Python 3.10
# numpy==1.24.4 + scikit-learn==1.7.2

from sklearn.datasets import fetch_openml
import numpy as np


def load_mnist(normalize=True, flatten=True, test_size=0.2, random_state=42):
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff"
    )

    if normalize:
        X = X / 255.0

    y = y.ravel().astype(np.int64)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    test_count = int(n_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist()
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)