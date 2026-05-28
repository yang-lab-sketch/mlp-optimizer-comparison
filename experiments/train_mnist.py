import sys
import os
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
import matplotlib.pyplot as plt

from data.load_mnist import load_mnist
from models.mlp import MLP
from losses.cross_entropy import CrossEntropyLoss
from utils.metrics import accuracy, confusion_matrix
from utils.plot import (
    plot_loss_curve,
    plot_accuracy_curve,
    plot_confusion_matrix
)

# =========================
# 训练配置
# =========================
EPOCHS = 20
BATCH_SIZE = 128
HIDDEN_DIM = 64
LR = 0.1
OPTIMIZER = "adam"   # sgd | momentum | adam

RESULT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# 加载数据
# =========================
X_train, X_test, y_train, y_test = load_mnist()

# ✅ load_mnist 已完成归一化
y_train = y_train.ravel().astype(np.int64)
y_test = y_test.ravel().astype(np.int64)

# =========================
# 模型 / Loss / Optimizer
# =========================
model = MLP(
    input_dim=784,
    hidden_dim=HIDDEN_DIM,
    output_dim=10
)

criterion = CrossEntropyLoss()

if OPTIMIZER == "sgd":
    from optimizers.sgd import SGD
    optimizer = SGD(lr=LR)

elif OPTIMIZER == "momentum":
    from optimizers.momentum import Momentum
    optimizer = Momentum(lr=LR, beta=0.9)

elif OPTIMIZER == "adam":
    from optimizers.Adam import Adam
    optimizer = Adam(lr=0.001)

# =========================
# 训练
# =========================
loss_history = []
acc_history = []

n_samples = y_train.shape[0]
n_batches = n_samples // BATCH_SIZE

for epoch in range(EPOCHS):
    indices = np.random.permutation(n_samples)
    X_shuf = X_train[indices]
    y_shuf = y_train[indices]

    epoch_loss = 0.0

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = start + BATCH_SIZE

        X_batch = X_shuf[start:end]
        y_batch = y_shuf[start:end]

        logits = model.forward(X_batch)
        loss, _ = criterion(logits, y_batch)
        epoch_loss += loss

        grads = model.backward(y_batch)

        params = {
            "W1": model.W1, "b1": model.b1,
            "W2": model.W2, "b2": model.b2
        }
        optimizer.step(params, grads)

    epoch_loss /= n_batches

    val_logits = model.forward(X_test)
    _, _ = criterion(val_logits, y_test)
    val_preds = model.predict(X_test)
    val_acc = accuracy(val_preds, y_test)

    loss_history.append(epoch_loss)
    acc_history.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# =========================
# 可视化
# =========================
plot_loss_curve(loss_history)
plt.savefig(os.path.join(RESULT_DIR, f"loss_{OPTIMIZER}.png"))
plt.show()
plt.close()

plot_accuracy_curve(acc_history)
plt.savefig(os.path.join(RESULT_DIR, f"accuracy_{OPTIMIZER}.png"))
plt.show()
plt.close()

final_preds = model.predict(X_test)
cm = confusion_matrix(final_preds, y_test)
plot_confusion_matrix(cm)
plt.savefig(os.path.join(RESULT_DIR, f"confusion_matrix_{OPTIMIZER}.png"))
plt.show()
plt.close()

# =========================
# 导出 Adam 训练结果
# =========================
if OPTIMIZER == "adam":
    save_path = os.path.join(RESULT_DIR, "mlp_adam.pth")
    np.savez(
        save_path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2
    )
    print(f"✅ Adam 模型已导出到 {save_path}")

print("✅ 训练完成，结果已保存到 results/")