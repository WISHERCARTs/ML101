from scipy.io import loadmat # Load the MNIST dataset
import matplotlib.pyplot as plt # Import matplotlib for visualization

mnist_raw = loadmat("mnist-original.mat") # Load the MNIST dataset
# print(mnist_raw)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}
x,y = mnist["data"],mnist["target"] # Split the data into features and labels

number = x[2000]
number_image = number.reshape(28,28) # Reshape the number to a 28x28 image

print(y[2000])
plt.imshow(
    number_image,
    cmap=plt.cm.binary, # Use binary color map
    interpolation="nearest") # Display the image
plt.show()
