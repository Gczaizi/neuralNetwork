import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

img_array = scipy.misc.imread('./4.png', flatten=True)

img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01

img_array = np.asfarray(img_data).reshape((28,28))

print(img_array)

plt.imshow(img_array, cmap='Greys', interpolation='None')
plt.show()