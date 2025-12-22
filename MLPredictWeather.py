# Linear Regression - ทำนายอุณหภูมิสูงสุด (MaxTemp) จากอุณหภูมิต่ำสุด (MinTemp)
import numpy as np  # ใช้จัดการ array
import pandas as pd  # ใช้จัดการข้อมูลแบบตาราง (DataFrame)
import matplotlib.pyplot as plt  # ใช้แสดงกราฟ
from sklearn.model_selection import train_test_split  # ใช้แบ่งข้อมูล train/test
from sklearn.linear_model import LinearRegression  # โมเดล Linear Regression

# โหลด dataset จาก URL (ข้อมูลสภาพอากาศ)
dataset = pd.read_csv('https://raw.githubusercontent.com/kongruksiamza/MachineLearning/refs/heads/master/Linear%20Regression/Weather.csv')

# เตรียมข้อมูล: แยก feature (x) และ target (y)
x = dataset["MinTemp"].values.reshape(-1, 1)  # MinTemp เป็น input (ต้อง reshape เป็น 2D array)
y = dataset["MaxTemp"].values.reshape(-1, 1)  # MaxTemp เป็น output ที่ต้องการทำนาย

# แบ่งข้อมูล 80% สำหรับ train, 20% สำหรับ test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# สร้างและ train โมเดล Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)  # เรียนรู้ความสัมพันธ์ระหว่าง MinTemp และ MaxTemp

# ทดสอบโมเดล: ใช้ x_test ทำนายค่า MaxTemp
y_pred = model.predict(x_test)

# เปรียบเทียบค่าจริง (Actual) กับค่าที่ทำนาย (Predicted)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())  # แสดง 5 แถวแรก
print(df.shape)  # แสดงขนาดของ DataFrame

# แสดงกราฟแท่งเปรียบเทียบ Actual vs Predicted (20 รายการแรก)
df1 = df.head(20)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title('Actual vs Predicted')
plt.show()

# --- (comment ไว้) ---
# แสดง scatter plot พร้อมเส้น regression
# plt.scatter(x_test, y_test, color='blue')  # จุดข้อมูลจริง
# plt.plot(x_test, y_pred, color='red', linewidth=2)  # เส้นทำนาย
# plt.show()

# print(dataset.shape)  # แสดงขนาด dataset

# แสดง scatter plot ของข้อมูลทั้งหมด
# dataset.plot(x='MinTemp', y='MaxTemp', style='o')
# plt.title('MinTemp vs MaxTemp')
# plt.xlabel('MinTemp')
# plt.ylabel('MaxTemp')
# plt.show()

# print(dataset.describe())  # แสดงสถิติเบื้องต้นของ dataset (mean, std, min, max, etc.) 