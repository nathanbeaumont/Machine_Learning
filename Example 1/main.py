import sys
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

print('Python %s on %s' % (sys.version, sys.platform))

# retrieve data set
mnist = fetch_openml('mnist_784', version=1, cache=True)

x, y = mnist["data"], mnist["target"]
print(x.shape)
print(y.shape)

some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

print(y[36000])

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)

x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
print(sgd_clf.predict([some_digit]))
