import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history, title="Loss Curve"):
    
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_accuracy_curve(acc_history, title="Accuracy Curve"):
    
    plt.figure(figsize=(6, 4))
    plt.plot(acc_history, marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_confusion_matrix(cm, classes=range(10)):
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # 标注数值
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()