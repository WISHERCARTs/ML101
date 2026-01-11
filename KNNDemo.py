"""
K-Nearest Neighbors (KNN) Classification - Iris Dataset
========================================================

สารบัญ (Table of Contents) - อ่านตามลำดับนี้:
   1. IMPORT LIBRARIES      - นำเข้า library
   2. LOAD DATA             - โหลดข้อมูล Iris
   3. SPLIT DATA            - แบ่ง Train/Test
   4. CREATE MODEL          - สร้างโมเดล KNN
   5. TRAIN MODEL           - train โมเดล
   6. PREDICT               - ทำนาย
   7. EVALUATE              - ประเมินผล

หลักการ KNN (K-Nearest Neighbors):
   - เป็นอัลกอริทึมแบบ "ดูเพื่อนบ้าน"
   - เมื่อได้ข้อมูลใหม่ จะหา K ตัวอย่างที่ใกล้ที่สุด (nearest neighbors)
   - แล้วโหวตว่าเพื่อนบ้านส่วนใหญ่เป็น class อะไร → ทำนายเป็น class นั้น
   
   ตัวอย่าง (K=3):
       ข้อมูลใหม่ → หา 3 เพื่อนบ้านใกล้สุด → [A, A, B] → โหวต A ชนะ → ทำนายเป็น A

   Parameter สำคัญ:
       - n_neighbors (K): จำนวนเพื่อนบ้านที่จะดู
         - K น้อย (เช่น 1): ไวต่อ noise, อาจ overfit
         - K มาก (เช่น 10): smooth กว่า, อาจ underfit

Dataset: Iris (ดอกไอริส 3 สายพันธุ์)
   - 150 ตัวอย่าง, 4 features (sepal/petal length & width)
   - 3 classes: setosa, versicolor, virginica
"""

# ========================== 1. IMPORT LIBRARIES ==========================
from sklearn.datasets import load_iris                # โหลด Iris dataset
from sklearn.model_selection import train_test_split  # แบ่ง train/test
from sklearn.neighbors import KNeighborsClassifier    # โมเดล KNN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # วัดผล


# ========================== 2. LOAD DATA ==========================
# โหลด Iris dataset (มี 150 ดอกไม้, 4 features, 3 classes)
iris_dataset = load_iris()

# iris_dataset ประกอบด้วย:
#   - 'data': ข้อมูล features (150, 4)
#   - 'target': label 0, 1, 2 (150,)
#   - 'target_names': ['setosa', 'versicolor', 'virginica']
#   - 'feature_names': ['sepal length', 'sepal width', 'petal length', 'petal width']


# ========================== 3. SPLIT DATA ==========================
# แบ่งข้อมูล: 60% train, 40% test
# test_size=0.4 หมายถึง 40% สำหรับ test
# random_state=0 เพื่อให้ผลลัพธ์เหมือนกันทุกครั้งที่รัน
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'],     # features (150, 4)
    iris_dataset['target'],   # labels (150,)
    test_size=0.4,            # 40% test, 60% train
    random_state=0            # seed สำหรับการสุ่ม
)

# ผลลัพธ์:
#   x_train: (90, 4)  - 90 ตัวอย่างสำหรับ train
#   x_test:  (60, 4)  - 60 ตัวอย่างสำหรับ test


# ========================== 4. CREATE MODEL ==========================
# สร้างโมเดล KNN
# n_neighbors=1 หมายถึง ดูเพื่อนบ้าน 1 คนที่ใกล้ที่สุด
# ลองเปลี่ยนเป็น 3, 5, 7 ดูว่า accuracy เปลี่ยนไหม
knn = KNeighborsClassifier(n_neighbors=1)


# ========================== 5. TRAIN MODEL ==========================
# train โมเดลด้วยข้อมูล training
# KNN จะจำข้อมูลทั้งหมดไว้ (ไม่ได้เรียนรู้ pattern แบบ Neural Network)
knn.fit(x_train, y_train)


# ========================== 6. PREDICT ==========================
# --- 6.1 ทำนายรูปเดี่ยว ---
# ทำนาย 1 ตัวอย่าง (x_test[1])
# ต้องใส่ [] ครอบเพราะ predict ต้องการ 2D array
pred = knn.predict([x_test[1]])

# แสดงผลการทำนายรูปเดี่ยว (uncomment เพื่อใช้)
# print("Prediction: ", pred)
# print("Actual: ", iris_dataset['target_names'][pred])

# --- 6.2 ทำนายทั้งหมด ---
# ทำนายทุกตัวอย่างใน test set
y_pred = knn.predict(x_test)


# ========================== 7. EVALUATE ==========================
# ประเมินประสิทธิภาพโมเดล

# --- 7.1 Classification Report (uncomment เพื่อใช้) ---
# แสดง precision, recall, f1-score ของแต่ละ class
# print(classification_report(y_test, y_pred, target_names=iris_dataset['target_names']))

# --- 7.2 Accuracy ---
# ความแม่นยำ = ทำนายถูกกี่ % จากทั้งหมด
print("ความแม่นยำของโมเดล =", accuracy_score(y_test, y_pred) * 100, "%")

# --- 7.3 ดูขนาดข้อมูล (uncomment เพื่อใช้) ---
# print(x_test.shape)  # (60, 4) -> 60 ตัวอย่าง, 4 features