# ==================== Face Recognition - แสดงรูปใบหน้า ====================
# จุดประสงค์: โหลด dataset รูปใบหน้าคนดัง (LFW) และแสดงภาพตัวอย่าง
# LFW = Labeled Faces in the Wild (รูปใบหน้าจริงจากอินเทอร์เน็ต)

# ==================== 1. Import Libraries ====================
from sklearn.datasets import fetch_lfw_people  # โหลด dataset ใบหน้าคนดัง
import matplotlib.pyplot as plt                 # สร้างกราฟและแสดงรูป

# ==================== 2. โหลด Dataset ====================
# fetch_lfw_people จะดาวน์โหลดรูปใบหน้าจาก internet (ครั้งแรกจะนาน)
# min_faces_per_person=60 : เลือกเฉพาะคนที่มีรูปอย่างน้อย 60 รูป
faces = fetch_lfw_people(min_faces_per_person=60)

# faces.target_names = รายชื่อคนทั้งหมด เช่น ['Ariel Sharon', 'Colin Powell', ...]
# faces.images.shape = (จำนวนรูป, สูง, กว้าง) เช่น (1348, 62, 47)
# faces.target = array ของ index ที่บอกว่ารูปนี้เป็นคนไหน [0, 1, 2, 0, 3, ...]

# ==================== 3. แสดงรูปตัวอย่าง ====================
# สร้าง grid 3 แถว x 5 คอลัมน์ = 15 รูป
fig, ax = plt.subplots(3, 5)

# วนลูปแสดงรูปทีละช่อง
# enumerate(ax.flat) = วนลูป subplot ทั้งหมดแบบ flat (ไม่สน row/col)
for i, axi in enumerate(ax.flat):
    # แสดงรูปใบหน้า (cmap='bone' = สีเทา-ฟ้าอ่อน)
    axi.imshow(faces.images[i], cmap='bone')
    
    # ลบแกน x, y (ไม่ต้องแสดงตัวเลข)
    axi.set(xticks=[], yticks=[])
    
    # แสดงชื่อคน: faces.target[i] = index ของคน, target_names[index] = ชื่อจริง
    axi.set_ylabel(faces.target_names[faces.target[i]], color='black')

plt.show()
# 💡 ขั้นตอนถัดไป: ใช้ PCA + SVM เพื่อจำแนกใบหน้าอัตโนมัติ