#Datasets แบ่งข้อมูลออกเป็น train และ test 75% train และ 25% test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
#(150,4)
#75% train และ 25% test
x_train,x_test,y_train,y_test = train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0) #random_state=0 คือ กำหนด seed เพื่อให้ผลลัพธ์การแบ่งข้อมูลเหมือนกันทุกครั้งที่รัน
# ถ้าต้องการใส่เพิ่ม test_size=0.2 คือ 20% test 80% train

print(x_train.shape) # (112,4) 112 คือจำนวนแถว 4 คือจำนวนคอลัมน์
print(y_train.shape) # (112,) 112 คือจำนวนแถว
print(x_test.shape) # (38,4) 38 คือจำนวนแถว
print(y_test.shape) # (38,) 38 คือจำนวนแถว


