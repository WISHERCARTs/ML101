# โหลด MNIST Dataset - ชุดข้อมูลตัวเลขที่เขียนด้วยมือ (0-9)
from scipy.io import loadmat  # ใช้โหลดไฟล์ .mat (ไฟล์ของ MATLAB)
import matplotlib.pyplot as plt  # ใช้แสดงรูปภาพ

# โหลดไฟล์ mnist-original.mat เข้ามา (70,000 รูป ขนาด 28x28 pixels)
mnist_raw = loadmat("mnist-original.mat")

# จัดรูปแบบข้อมูลใหม่ให้ใช้งานง่าย
mnist = {
    "data": mnist_raw["data"].T,  # .T = transpose สลับแถว-คอลัมน์ ให้แต่ละแถวเป็น 1 รูป
    "target": mnist_raw["label"][0]  # label = ตัวเลขที่ถูกต้อง (0-9)
}

# แยกข้อมูลออกเป็น x (รูปภาพ) และ y (คำตอบ)
x, y = mnist["data"], mnist["target"]

# เลือกรูปที่ index 2000 มาแสดง
number = x[2000]  # ดึงข้อมูล pixel ของรูปที่ 2000 (มี 784 pixels)
number_image = number.reshape(28, 28)  # แปลงจาก 1D array (784,) เป็น 2D array (28x28)

# แสดงผลลัพธ์
print(y[2000])  # พิมพ์ว่ารูปนี้เป็นเลขอะไร
plt.imshow(
    number_image,
    cmap=plt.cm.binary,  # ใช้สีขาว-ดำ คือ 0 คือขาว 1 คือดำ _r คือ reverse คือ 0 คือดำ 1 คือขาว
    interpolation="nearest"  # แสดง pixel ตรงๆ ไม่ blur
)
plt.show()  # แสดงรูปภาพ
