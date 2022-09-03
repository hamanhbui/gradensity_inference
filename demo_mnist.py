import gzip

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.neural_network import MLPClassifier
import sklearn.metrics

image_size = 28
df_orig_train = pd.read_csv("/home/ubuntu/datasets/MNIST/raw/mnist_train_final.csv")
df_orig_test = pd.read_csv("/home/ubuntu/datasets/MNIST/raw//mnist_test_final.csv")

labels_train = df_orig_train["label"]
df_train_digits = df_orig_train.drop("label", axis=1)

labels_test = df_orig_test["label"]
df_test_digits = df_orig_test.drop("label", axis=1)

# Train
train_images = []
train_labels = []

for index, row in df_train_digits.iterrows():
    if index > 9999:
        break

    image = df_train_digits.iloc[index].to_numpy()/255
    label = labels_train[index]

    train_images.append(image)
    train_labels.append(label)

x_tr = np.array(train_images)
y_tr = np.array(train_labels)

# Test
test_images = []
test_labels = []

for index, row in df_test_digits.iterrows():
    image = df_test_digits.iloc[index].to_numpy()/255
    label = labels_test[index]

    test_images.append(image)
    test_labels.append(label)

x_te = np.array(test_images)
y_te = np.array(test_labels)

# model = MLPClassifier(hidden_layer_sizes=(40,),
#     max_iter=8,
#     alpha=1e-4,
#     solver="sgd",
#     verbose=10,
#     random_state=1,
#     learning_rate_init=0.2)

model = MLPClassifier(hidden_layer_sizes = (5,2), solver = 'sgd', learning_rate_init = 0.001, max_iter = 1000)
model.fit(x_tr, y_tr)

y_hat = model.predict(x_tr)
print(sklearn.metrics.accuracy_score(y_hat, y_tr))

y_hat = model.predict(x_te)
print(sklearn.metrics.accuracy_score(y_hat, y_te))