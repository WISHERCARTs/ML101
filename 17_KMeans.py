# ==================== K-Means Clustering ====================
# จุดประสงค์: จัดกลุ่มข้อมูลแบบ Unsupervised (ไม่ต้องบอกคำตอบ) 
# K-Means จะหา "จุดศูนย์กลาง" (Centroid) ของแต่ละกลุ่มเอง

# ==================== 1. Import Libraries ====================
from sklearn.datasets import make_blobs  # สร้างข้อมูลตัวอย่างเป็นกลุ่มๆ
from sklearn.cluster import KMeans       # K-Means Algorithm
import matplotlib.pyplot as plt          # สร้างกราฟและแสดงรูป

# ==================== 2. สร้างข้อมูลตัวอย่าง ====================
# make_blobs สร้างข้อมูลจุดที่กระจายเป็นกลุ่มๆ
# n_samples=300  : จำนวนจุดทั้งหมด 300 จุด
# centers=4      : แบ่งเป็น 4 กลุ่ม
# cluster_std=0.5: ความกระจายของแต่ละกลุ่ม (ยิ่งน้อยยิ่งแน่น)
x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

# สร้างจุดใหม่ 10 จุด สำหรับทดสอบ prediction
x_test, y_test = make_blobs(n_samples=10, centers=4, cluster_std=0.5, random_state=0)

# ==================== 3. Train K-Means Model ====================
# n_clusters=4 : กำหนดให้แบ่งเป็น 4 กลุ่ม
model = KMeans(n_clusters=4, random_state=0)
model.fit(x)  # เทรนโมเดล: หา centroid ของแต่ละกลุ่ม

# ==================== 4. Predict ====================
y_pred = model.predict(x)            # ทำนายว่าแต่ละจุดอยู่กลุ่มไหน (0, 1, 2, 3)
y_pred_new = model.predict(x_test)   # ทำนายกลุ่มของจุดใหม่
center = model.cluster_centers_      # พิกัดจุดศูนย์กลางของแต่ละกลุ่ม [x, y]

# ==================== 5. แสดงผลกราฟ ====================
# วาดจุดข้อมูลทั้งหมด (สีตามกลุ่ม)
# x[:,0] = ค่า x ของทุกจุด, x[:,1] = ค่า y ของทุกจุด
plt.scatter(x[:, 0], x[:, 1], c=y_pred)          # ข้อมูล train (จุดเล็ก)
plt.scatter(x_test[:, 0], x_test[:, 1], s=100, c=y_pred_new)  # ข้อมูล test (จุดใหญ่)

# วาด Centroid (จุดศูนย์กลาง) ของแต่ละกลุ่ม
plt.scatter(center[0, 0], center[0, 1], c='blue', label='centroid 1')
plt.scatter(center[1, 0], center[1, 1], c='green', label='centroid 2')
plt.scatter(center[2, 0], center[2, 1], c='red', label='centroid 3')
plt.scatter(center[3, 0], center[3, 1], c='black', label='centroid 4')

plt.legend(frameon=True)  # แสดง legend (กรอบชื่อ centroid)
plt.show()
# 💡 K-Means ทำงาน: หา centroid → จับจุดใกล้ centroid ไหนก็อยู่กลุ่มนั้น