from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.io import loadmat


# Load dataset
mnist_raw = loadmat("mnist-original.mat")
mnist = {
    'data': mnist_raw['data'].T,
    'target': mnist_raw['label'][0]
}
x_train,y_train,x_test,y_test = train_test_split(mnist['data'],mnist['target'],test_size=0.2,random_state=0)

print("Before PCA:",x_train.shape)
#พอ print ออกมาจะเห็นว่ามันมี 784 features ซึ่งมันเยอะเกินไปเราเลยจะใช้ PCA ลดมิติลงเหลือ 2 features

pca = PCA(.80)
data = pca.fit_transform(x_train)
result = pca.inverse_transform(data)

print("After PCA:",data.shape)
#มันจะลดมิติลงเหลือ 54 features ซึ่งมันก็ยังเยอะอยู่ดีเราเลยจะใช้ PCA ลดมิติลงเหลือ 2 features
print(pca.n_components_)

#show image
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
#image features 784
plt.imshow(mnist['data'][0].reshape(28,28),cmap='gray',interpolation='nearest')
plt.xlabel('784 features')
plt.title("Original")   

plt.subplot(1,2,2)
#image features 95% -> 54
plt.imshow(result[0].reshape(28,28),cmap='gray',interpolation='nearest')
plt.xlabel('54 features')
plt.title("PCA")
plt.show()

