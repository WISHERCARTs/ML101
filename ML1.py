# โหลด Digits Dataset จาก sklearn - ชุดข้อมูลตัวเลขที่เขียนด้วยมือ (0-9) ขนาด 8x8 pixels
import pylab  # ใช้แสดงรูปภาพ (เป็น alias ของ matplotlib)
from sklearn import datasets  # ใช้โหลด dataset ที่มีมาให้ใน sklearn

# โหลด dataset ตัวเลข (มี 1,797 รูป ขนาด 8x8 pixels)
digits_dataset = datasets.load_digits()

# พิมพ์ label ของ 10 รูปแรก ว่าแต่ละรูปเป็นเลขอะไร
print(digits_dataset.target[:10])
# pylab.imshow(digits_dataset.images[:10], cmap=pylab.cm.gray_r, interpolation="nearest") # แสดงรูปภาพ 10 รูปแรก 

# แสดงรูปภาพ 10 รูปแรกใน grid 2 แถว x 5 คอลัมน์
for i in range(10):
    pylab.subplot(2, 5, i + 1)  # สร้างช่องที่ i+1 ใน grid 2x5
    pylab.imshow(
        digits_dataset.images[i],  # รูปภาพที่ i
        cmap=pylab.cm.gray_r,  # ใช้สี grayscale แบบ reverse (พื้นขาว ตัวเลขดำ)
        interpolation="nearest"  # แสดง pixel ตรงๆ ไม่ blur
    )
    pylab.title(digits_dataset.target[i])  # แสดงตัวเลขที่ถูกต้องเป็น title
pylab.show()  # แสดงรูปทั้งหมด
