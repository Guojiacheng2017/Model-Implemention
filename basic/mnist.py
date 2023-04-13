import numpy as np
# from keras.datasets import mnist
from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt

mnist = load_digits()
# (train_X, train_y), (test_X, test_y) = mnist.load_data()
# print(pd.DataFrame(mnist.data).head())

fig, axes = plt.subplots(2, 10, figsize=(16, 6))
for i in range(20):
    axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray');
    axes[i // 10, i % 10].axis('off')
    axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")

plt.tight_layout()
