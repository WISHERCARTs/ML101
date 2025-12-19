# import matplotlib.pyplot as pylab
import pylab
from sklearn import datasets
digits_dataset = datasets.load_digits()

print(digits_dataset.target[:10])
# pylab.imshow(digits_dataset.images[:10], cmap=pylab.cm.gray_r, interpolation="nearest")
for i in range(10):
    pylab.subplot(2, 5, i + 1)
    pylab.imshow(digits_dataset.images[i], cmap=pylab.cm.gray_r, interpolation="nearest")
    pylab.title(digits_dataset.target[i])
pylab.show()
