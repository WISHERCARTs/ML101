"""
ML5.py - Linear Equation Graph (สมการเส้นตรง)
วัตถุประสงค์: วาดกราฟสมการเส้นตรง y = ax + b
             เพื่อเข้าใจพื้นฐานก่อนเรียน Linear Regression
"""

# Import Libraries
import numpy as np              # คำนวณตัวเลข
import matplotlib.pyplot as plt # วาดกราฟ

# สมการเส้นตรง: y = ax + b
# - a = ความชัน (slope) -> ถ้า a > 0 เส้นขึ้น, a < 0 เส้นลง
# - b = จุดตัดแกน Y (y-intercept) -> เส้นตัดแกน Y ที่จุดนี้

# สร้างข้อมูล
x = np.linspace(-5, 5, 100)  # สร้าง 100 ค่า ตั้งแต่ -5 ถึง 5
y = 2*x + 1                   # y = 2x + 1 → ความชัน=2, ตัดแกน Y ที่ 1

# วาดกราฟ
# '-r' = เส้นทึบ(-) สีแดง(r)
# อื่นๆ: '--b'=เส้นประสีน้ำเงิน, ':g'=จุดสีเขียว, 'ok'=วงกลมสีดำ
plt.plot(x, y, '-r', label='y=2x+1')

# ตกแต่งกราฟ
plt.xlabel('x')                  # ชื่อแกน X
plt.ylabel('y')                  # ชื่อแกน Y
plt.legend(loc='upper left')     # แสดง legend มุมซ้ายบน
plt.title("Graph y=2x+1")        # ชื่อกราฟ
plt.grid()                       # แสดงเส้น grid

plt.show()