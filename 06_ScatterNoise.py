"""
ML6.py - Scatter Plot with Random Noise
วัตถุประสงค์: จำลองข้อมูลจริงที่มี noise (ความคลาดเคลื่อน)
             เพื่อเตรียมเรียน Linear Regression
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลแบบสุ่ม
rng = np.random                # ตัวสร้างเลขสุ่ม

# rand(50) = สุ่ม 50 ค่า ระหว่าง 0-1, แล้ว *10 ให้เป็น 0-10
x = rng.rand(50) * 10

# randn(50) = สุ่มแบบ Normal Distribution (ค่าเฉลี่ย=0, SD=1)
# ทำให้ y ไม่ได้อยู่บนเส้นตรงพอดี แต่มี "noise" กระจายรอบเส้น
y = 2*x + rng.randn(50)

# วาด Scatter Plot
# ต่างจาก plt.plot() ตรงที่ไม่เชื่อมจุด แสดงเป็นจุดๆ
plt.scatter(x, y)

# ตกแต่งกราฟ
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.title("Scatter Plot")
plt.legend(["y = 2x + noise"])

plt.show()