"""
=============================================================================
📊 KNN (K-Nearest Neighbors) - Diabetes Prediction
=============================================================================
เป้าหมาย: ทำนายว่าคนไข้จะเป็นเบาหวานหรือไม่ โดยใช้ข้อมูลสุขภาพ
Dataset: Pima Indians Diabetes Database (768 คน, 8 features)

Features ที่ใช้ทำนาย:
- Pregnancies (จำนวนครั้งที่ตั้งครรภ์)
- Glucose (ระดับน้ำตาลในเลือด)
- BloodPressure (ความดันโลหิต)
- SkinThickness (ความหนาผิวหนัง)
- Insulin (ระดับอินซูลิน)
- BMI (ดัชนีมวลกาย)
- DiabetesPedigreeFunction (ประวัติเบาหวานในครอบครัว)
- Age (อายุ)

Target (สิ่งที่ต้องการทำนาย):
- Outcome: 0 = ไม่เป็นเบาหวาน, 1 = เป็นเบาหวาน
=============================================================================
"""

# =========================
# 📦 STEP 1: Import Libraries
# =========================
from sklearn.model_selection import train_test_split  # แบ่งข้อมูล train/test
from sklearn.neighbors import KNeighborsClassifier    # KNN algorithm
from sklearn.metrics import classification_report, confusion_matrix # accuracy, precision, recall, f1-score, confusion matrix
import kagglehub      # ดาวน์โหลด dataset จาก Kaggle
import pandas as pd   # จัดการข้อมูลตาราง
import os             # จัดการ file path
import numpy as np    # คำนวณตัวเลข
import matplotlib.pyplot as plt  # สร้างกราฟ

# =========================
# 📥 STEP 2: Download Dataset
# =========================
# ดาวน์โหลด Pima Indians Diabetes dataset จาก Kaggle
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)

# สร้าง path ไปยังไฟล์ CSV
csv_file = os.path.join(path, "diabetes.csv")

# =========================
# 📖 STEP 3: Load & Prepare Data
# =========================
# อ่านข้อมูลจาก CSV เข้ามาเป็น DataFrame
df = pd.read_csv(csv_file)

# แยก Features (x) และ Target (y)
# x = ข้อมูลที่ใช้ทำนาย (8 columns: Pregnancies, Glucose, etc.)
# drop("Outcome") = เอา column Outcome ออก, เหลือแต่ features
x = df.drop("Outcome", axis=1).values

# y = ค่าที่ต้องการทำนาย (0 = ไม่เป็นเบาหวาน, 1 = เป็นเบาหวาน)
y = df["Outcome"].values

# =========================
# ✂️ STEP 4: Split Data (Train/Test)
# =========================
# แบ่งข้อมูลเป็น 2 ชุด:
# - 60% สำหรับ train (สอน model)
# - 40% สำหรับ test (ทดสอบความแม่นยำ)
# random_state=42 ทำให้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# =========================
# 🔍 STEP 5: Find Best K Value
# =========================
# KNN ต้องเลือกค่า K (จำนวน neighbors ที่ใช้ vote)
# K ต่ำเกินไป = overfit (จำข้อมูล train ดีมาก แต่ทำนาย test ไม่ดี)
# K สูงเกินไป = underfit (ทำนายไม่ค่อยแม่นยำ)
# เราจะลอง k = 1, 2, 3, ..., 8 เพื่อหาค่าที่ดีที่สุด

k_neighbor = np.arange(1, 9)  # สร้าง array [1, 2, 3, 4, 5, 6, 7, 8]

# สร้าง array ว่างเพื่อเก็บ score ของแต่ละ k
train_score = np.empty(len(k_neighbor))  # เก็บ accuracy บน train data
test_score = np.empty(len(k_neighbor))   # เก็บ accuracy บน test data

# วนลูปทดสอบแต่ละค่า k
for i, k in enumerate(k_neighbor):
    # enumerate(k_neighbor) จะ return: (0,1), (1,2), (2,3), ...
    # i = index (ตำแหน่งใน array)
    # k = ค่า k ที่จะทดสอบ
    
    # สร้าง KNN model ด้วย k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # สอน model ด้วย training data
    knn.fit(x_train, y_train)
    
    # วัดความแม่นยำ (accuracy)
    # .score() return ค่า 0-1 (1 = แม่นยำ 100%)
    train_score[i] = knn.score(x_train, y_train)  # ทดสอบกับ train data
    test_score[i] = knn.score(x_test, y_test)     # ทดสอบกับ test data
    
    # แสดงผล test accuracy เป็น %
    print(f"k={k}: Test Accuracy = {test_score[i]*100:.2f}%")

# =========================
# 📈 STEP 6: Plot Results
# =========================
# สร้างกราฟเปรียบเทียบ train vs test accuracy
plt.plot(k_neighbor, train_score, label="Train Score")  # เส้น train
plt.plot(k_neighbor, test_score, label="Test Score")    # เส้น test
# หมายเหตุ:
# - Train Score สูง = model จำข้อมูลที่เคยเห็นได้ดี
# - Test Score สูง = model ทำนายข้อมูลใหม่ได้ดี (อันนี้สำคัญกว่า!)
# - ถ้า Train สูงมาก แต่ Test ต่ำ = Overfitting ❌

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy: Finding Best K Value")
plt.legend()
plt.show()

# =========================
# 💡 สรุป: วิธีอ่านกราฟ
# =========================
# 1. ดูที่เส้น Test Score (สีส้ม) เป็นหลัก
# 2. เลือก K ที่ทำให้ Test Score สูงที่สุด
# 3. ถ้า Train และ Test ใกล้กัน = model ดี (ไม่ overfit)

# print(y_test.shape)
# print(df.head())
# print(df.shape) # (768, 9) = 768 rows, 9 columns