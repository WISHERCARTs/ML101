from tornado.util import import_object
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import kagglehub
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "diabetes.csv")

#read data
df = pd.read_csv(csv_file)
# data
x = df.drop("Outcome", axis=1).values
# outcome data
y = df["Outcome"].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)

#find k to model
k_neighbor = np.arange(1,9)
#empty array
train_score = np.empty(len(k_neighbor))
test_score = np.empty(len(k_neighbor))

for i,k in enumerate(k_neighbor): #enumerate คือการนับเลขไปเรื่อยๆ
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(x_train,y_train)
    #train & test score
    train_score[i] = knn.score(x_train,y_train)
    test_score[i] = knn.score(x_test,y_test)
    print(test_score[i]*100)

plt.plot(k_neighbor,train_score,label="Train Score") #train score คือข้อมูลที่ model เคยเห็น
plt.plot(k_neighbor,test_score,label="Test Score") #test score คือข้อมูลที่ model ไม่เคยเห็น
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy")
plt.legend()
plt.show()

# print(y_test.shape)
# print(df.head())
# print(df.shape) # (768, 9)* #แสดงรูปร่างของข้อมูล