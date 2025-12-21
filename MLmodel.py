"""
MLmodel.py - Linear Regression Model
วัตถุประสงค์: สร้าง model หาเส้นตรง y = ax + b ที่ fit กับข้อมูลที่สุด
             - ใส่ข้อมูล (x, y) → model เรียนรู้หา a, b
             - ใช้ model ทำนายค่า y ใหม่จาก x ที่ไม่เคยเห็น
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random

# ========== 1. สร้างข้อมูลจำลอง ==========
x = rng.rand(50) * 10         # X: 50 ค่าสุ่ม (0-10)
y = 2*x + rng.randn(50)       # Y = 2x + noise (สมการจริงคือ y=2x)

# sklearn ต้องการ X เป็น 2D array (แถว x คอลัมน์)
# reshape(-1, 1) = แปลงจาก [1,2,3] → [[1],[2],[3]]
x_train = x.reshape(-1, 1)

# ========== 2. Train Model ==========
model = LinearRegression()
model.fit(x_train, y)         # Model เรียนรู้หา a, b จากข้อมูล

# ดูค่าที่ model เรียนรู้ได้:
# print(model.coef_[0])       # a = ความชัน (ควรได้ ~2)
# print(model.intercept_)     # b = จุดตัดแกน Y (ควรได้ ~0)
# print(model.score(x_train, y))  # R² score (0-1, ยิ่งใกล้ 1 ยิ่งดี)

# ========== 3. ทำนาย (Predict) ==========
# สร้างข้อมูล x ใหม่ที่ต้องการทำนาย
# -1 ถึง 11 เพื่อให้เส้นยาวกว่าข้อมูลเดิม (0-10) นิดหน่อย
xfit = np.linspace(-1, 11, 100)
xfit = xfit.reshape(-1, 1)

yfit = model.predict(xfit)    # ใช้ model ทำนาย y จาก x ใหม่

# ========== 4. แสดงผล ==========
plt.scatter(x, y, label='ข้อมูลจริง')           # จุดข้อมูล
plt.plot(xfit, yfit, 'r', label='เส้น Regression')  # เส้นที่ model หาได้
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()