# ==================== PCA กับ MNIST Dataset ====================
# จุดประสงค์: ใช้ PCA ลดมิติข้อมูลรูปภาพตัวเลข แล้วเปรียบเทียบภาพก่อน-หลัง
# MNIST = ชุดข้อมูลรูปภาพตัวเลข 0-9 ขนาด 28x28 (784 pixels)

# ==================== 1. Import Libraries ====================
from sklearn.model_selection import train_test_split  # แบ่งข้อมูล train/test
from sklearn.decomposition import PCA                  # ลดมิติข้อมูลด้วย PCA
import matplotlib.pyplot as plt                        # สร้างกราฟและแสดงรูป
from scipy.io import loadmat                           # อ่านไฟล์ .mat (MATLAB)


# ==================== 2. Load Dataset ====================
# โหลดไฟล์ MNIST จากไฟล์ .mat
mnist_raw = loadmat("mnist-original.mat")

# จัดรูปแบบข้อมูลให้ใช้งานง่าย
mnist = {
    'data': mnist_raw['data'].T,    # .T = transpose, ให้แต่ละแถวคือ 1 รูป (784 features)
    'target': mnist_raw['label'][0]  # ตัวเลขที่ถูกต้อง 0-9
}

# แบ่งข้อมูล 80% train, 20% test
x_train, y_train, x_test, y_test = train_test_split(
    mnist['data'], mnist['target'], 
    test_size=0.2, random_state=0
)

# ==================== 3. PCA - ลดมิติข้อมูล ====================
print("Before PCA:", x_train.shape)
# ผลลัพธ์: (56000, 784) = 56000 รูป, แต่ละรูปมี 784 features (28x28 pixels)
# 784 features เยอะเกินไป จึงใช้ PCA ลดมิติลง

# PCA(.80) = เก็บ variance ไว้ 80% (ข้อมูลสำคัญ 80%)
# PCA จะเลือกจำนวน components ที่เหมาะสมให้อัตโนมัติ
pca = PCA(.80)
data = pca.fit_transform(x_train)     # ลดมิติข้อมูล: 784 → ~43 features
result = pca.inverse_transform(data)  # แปลงกลับเป็น 784 features (สำหรับแสดงรูป)

print("After PCA:", data.shape)
# ผลลัพธ์: (56000, ~43) = ลดจาก 784 เหลือ ~43 features แต่ยังเก็บข้อมูล 80%
print("จำนวน Components:", pca.n_components_)

# ==================== 4. แสดงรูปเปรียบเทียบ ====================
plt.figure(figsize=(8, 4))  # กำหนดขนาดรูป

# --- รูปซ้าย: ภาพต้นฉบับ ---
plt.subplot(1, 2, 1)  # แบ่ง 1 แถว 2 คอลัมน์, ตำแหน่งที่ 1
# reshape(28,28) = เปลี่ยนจาก array 784 ช่อง เป็นรูป 28x28
plt.imshow(mnist['data'][0].reshape(28, 28), cmap='gray', interpolation='nearest')
plt.xlabel('784 features')  # แสดงจำนวน features เดิม
plt.title("Original")

# --- รูปขวา: ภาพหลัง PCA ---
plt.subplot(1, 2, 2)  # ตำแหน่งที่ 2
# result = ข้อมูลที่ถูกบีบอัดแล้วแปลงกลับ (สูญเสียข้อมูลบางส่วน)
plt.imshow(result[0].reshape(28, 28), cmap='gray', interpolation='nearest')
plt.xlabel('~43 features → 784')  # บีบอัดเหลือ ~43 แล้วแปลงกลับ
plt.title("PCA (80% variance)")

plt.show()
# 💡 สังเกต: ภาพหลัง PCA จะเบลอกว่าเล็กน้อย เพราะสูญเสียข้อมูล 20%

