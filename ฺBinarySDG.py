"""
Binary Classification ด้วย Stochastic Gradient Descent (SGD)
=============================================================

วัตถุประสงค์:
   - สร้างโมเดลที่ทำนายว่ารูปตัวเลขที่ให้มานั้น "ใช่เลข 5" หรือ "ไม่ใช่เลข 5"
   - เป็นการจำแนกแบบ Binary (2 คลาส): True = ใช่เลข 5, False = ไม่ใช่เลข 5

หลักการ Binary Classification:
   - ต่างจาก Multi-class ที่ทำนาย 0-9 (10 คลาส)
   - Binary ทำนายแค่ 2 คลาส: "ใช่" หรือ "ไม่ใช่"
   - ง่ายกว่าและเร็วกว่า เหมาะกับปัญหาแบบ Yes/No

หลักการ SGD (Stochastic Gradient Descent):
   - เป็น optimizer ที่ใช้หาค่า weight ที่ดีที่สุดสำหรับโมเดล
   - "Stochastic" = สุ่มเลือกข้อมูลทีละตัว (หรือ mini-batch) มา update weight
   - ข้อดี: เร็ว, ใช้ memory น้อย, เหมาะกับ dataset ขนาดใหญ่
   - ข้อเสีย: ผลลัพธ์อาจ fluctuate (ไม่นิ่ง) เพราะสุ่ม

Dataset: MNIST (70,000 รูปตัวเลข 0-9 ขนาด 28x28 pixels)
   - Training: 60,000 รูป (ใช้สอนโมเดล)
   - Testing: 10,000 รูป (ใช้ทดสอบโมเดล)
"""

# ========================== 1. IMPORT LIBRARIES ==========================
from scipy.io import loadmat          # โหลดไฟล์ .mat (format ของ MATLAB)
import numpy as np                     # จัดการ array
import matplotlib.pyplot as plt        # แสดงรูปภาพ
from sklearn.linear_model import SGDClassifier  # โมเดล SGD สำหรับ Classification


# ========================== 2. HELPER FUNCTIONS ==========================

def displayImage(x):
    """
    แสดงรูปภาพตัวเลขจาก MNIST dataset
    
    Parameters:
        x: array ขนาด (784,) = 28*28 pixels ที่ถูก flatten เป็น 1D
    
    การทำงาน:
        1. reshape จาก (784,) กลับเป็น (28, 28) เพื่อแสดงเป็นรูป
        2. ใช้ colormap 'hot' (ดำ→แดง→เหลือง→ขาว)
        3. interpolation='nearest' = แสดง pixel ตรงๆ ไม่ blur
    """
    plt.imshow(
        x.reshape(28, 28),          # แปลงจาก 1D (784,) → 2D (28x28)
        cmap=plt.cm.hot,            # สี: พื้นดำ, ตัวเลขสีส้ม/เหลือง
        interpolation="nearest"     # แสดง pixel คมชัด ไม่เบลอ
    )
    plt.axis("off")                 # ซ่อนแกน x, y
    plt.show()


def displayPredict(clf, actually_y, x):
    """
    แสดงผลการทำนายเทียบกับค่าจริง
    
    Parameters:
        clf: โมเดลที่ train แล้ว
        actually_y: ค่าจริง (True = เป็นเลข 5, False = ไม่ใช่เลข 5)
        x: รูปภาพที่ต้องการทำนาย (784 pixels)
    
    Output:
        - Actually: ค่าจริงว่าใช่เลข 5 หรือไม่
        - Prediction: ค่าที่โมเดลทำนาย
        - ถ้าทั้งสองค่าตรงกัน = โมเดลทำนายถูก!
    """
    print("=" * 40)
    print(f"Actually   : {actually_y}")    # ค่าจริง
    print(f"Prediction : {clf.predict([x])[0]}")  # ค่าที่โมเดลทำนาย
    # clf.predict([x]) return array เช่น [True] ต้องใส่ [0] เพื่อเอาค่าออกมา
    print("=" * 40)


# ========================== 3. LOAD DATA ==========================
# โหลด MNIST dataset จากไฟล์ .mat
# ไฟล์นี้มี 70,000 รูปตัวเลข 0-9

mnist_raw = loadmat('mnist-original.mat')

# จัดรูปแบบข้อมูลให้ใช้งานง่าย
mnist = {
    'data': mnist_raw["data"].T,    # .T = transpose ให้แต่ละแถวเป็น 1 รูป (70000, 784)
    'target': mnist_raw["label"][0]  # label 0-9 บอกว่าแต่ละรูปเป็นเลขอะไร (70000,)
}

# แยกข้อมูลเป็น x (รูปภาพ) และ y (label)
x, y = mnist["data"], mnist["target"]

# ตรวจสอบขนาดข้อมูล (uncomment เพื่อดู)
# print(f"Total images: {x.shape}")    # (70000, 784) = 70000 รูป, 784 pixels/รูป
# print(f"Total labels: {y.shape}")    # (70000,) = 70000 labels


# ========================== 4. SPLIT DATA ==========================
# แบ่งข้อมูล MNIST ตามมาตรฐาน:
#   - Training: 60,000 รูปแรก (index 0-59999)
#   - Testing: 10,000 รูปหลัง (index 60000-69999)
#
# หมายเหตุ: ไม่ใช้ train_test_split เพราะ MNIST มีมาตรฐานแบ่งอยู่แล้ว

x_train = x[:60000]     # รูปสำหรับ training (60000, 784)
x_test = x[60000:]      # รูปสำหรับ testing (10000, 784)
y_train = y[:60000]     # label สำหรับ training (60000,) ค่า 0-9
y_test = y[60000:]      # label สำหรับ testing (10000,) ค่า 0-9

# ตรวจสอบขนาด (uncomment เพื่อดู)
# print(f"Training: {x_train.shape}, {y_train.shape}")  # (60000, 784), (60000,)
# print(f"Testing: {x_test.shape}, {y_test.shape}")     # (10000, 784), (10000,)


# ========================== 5. PREPARE BINARY LABELS ==========================
# แปลง label จาก 0-9 เป็น True/False สำหรับ Binary Classification
#
# ตัวอย่าง: ทำนายว่า "รูปนี้เป็นเลข 5 หรือไม่?"
#   - y_train = [0, 5, 3, 5, 2, 5, ...]
#   - y_train_5 = [False, True, False, True, False, True, ...]
#
# เปลี่ยนเลข 5 เป็นเลขอื่นได้ เช่น (y_train == 0) สำหรับทำนายเลข 0

TARGET_DIGIT = 5  # เลขที่ต้องการทำนาย (เปลี่ยนเป็น 0-9 ได้)

y_train_5 = (y_train == TARGET_DIGIT)  # True ถ้าเป็นเลข 5, False ถ้าไม่ใช่
y_test_5 = (y_test == TARGET_DIGIT)

# ตัวอย่างผลลัพธ์ (uncomment เพื่อดู)
# print(f"y_train[:10] = {y_train[:10]}")        # [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
# print(f"y_train_5[:10] = {y_train_5[:10]}")    # [True, False, False, ...]


# ========================== 6. CREATE & TRAIN MODEL ==========================
# สร้างโมเดล SGDClassifier และ train ด้วยข้อมูล
#
# SGDClassifier ทำงานอย่างไร:
#   1. เริ่มจาก weight สุ่ม
#   2. สุ่มเลือกข้อมูลทีละตัว (stochastic)
#   3. คำนวณ error ระหว่างค่าจริงกับค่าทำนาย
#   4. ปรับ weight ให้ error ลดลง (gradient descent)
#   5. ทำซ้ำจนครบทุกข้อมูล
#
# Default: ใช้ loss='hinge' (SVM) แต่เปลี่ยนเป็น 'log_loss' ได้ (Logistic Regression)

sgd_clf = SGDClassifier(
    random_state=42,    # seed สำหรับการสุ่ม (ให้ผลลัพธ์ reproducible)
    max_iter=1000,      # จำนวนรอบสูงสุดในการ train
    tol=1e-3            # หยุดถ้า loss ลดลงน้อยกว่านี้
)

sgd_clf.fit(x_train, y_train_5)  # Train โมเดล!
# โมเดลเรียนรู้ว่า pixel pattern แบบไหนเป็นเลข 5


# ========================== 7. TEST & VISUALIZE ==========================
# เลือกรูปจาก test set มาทดสอบ

predict_number = 5500  # index ของรูปที่ต้องการทดสอบ (0-9999)

# แสดงรูปภาพ
print(f"\nทดสอบรูปที่ index {predict_number}:")
print(f"รูปนี้คือเลข: {y_test[predict_number]:.0f}")  # label จริงว่าเป็นเลขอะไร
displayImage(x_test[predict_number])

# แสดงผลการทำนาย
displayPredict(sgd_clf, y_test_5[predict_number], x_test[predict_number])


# ========================== OPTIONAL: TEST MORE SAMPLES ==========================
# ทดสอบหลายรูปพร้อมกัน (uncomment เพื่อใช้)

# test_indices = [0, 100, 500, 1000, 5000]  # รูปที่ต้องการทดสอบ
# for idx in test_indices:
#     print(f"\n--- Test image {idx} ---")
#     print(f"Actual digit: {y_test[idx]:.0f}")
#     displayImage(x_test[idx])
#     displayPredict(sgd_clf, y_test_5[idx], x_test[idx])


# ========================== OPTIONAL: MODEL ACCURACY ==========================
# วัดความแม่นยำของโมเดล (uncomment เพื่อใช้)

# from sklearn.metrics import accuracy_score, confusion_matrix
# 
# # ทำนายทุกรูปใน test set
# y_pred = sgd_clf.predict(x_test)
# 
# # คำนวณ accuracy
# accuracy = accuracy_score(y_test_5, y_pred)
# print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
# 
# # แสดง Confusion Matrix
# cm = confusion_matrix(y_test_5, y_pred)
# print(f"\nConfusion Matrix:")
# print(f"              Predicted")
# print(f"              Not 5  | Is 5")
# print(f"Actual Not 5:  {cm[0][0]:5d} | {cm[0][1]:5d}")
# print(f"Actual Is 5:   {cm[1][0]:5d} | {cm[1][1]:5d}")