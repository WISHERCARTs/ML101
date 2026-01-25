"""
=============================================================================
Naive Bayes Classification บน Adult Dataset
=============================================================================
วัตถุประสงค์: ทำนายว่าคนคนหนึ่งจะมีรายได้ >50K หรือ <=50K ต่อปี
โดยใช้ Naive Bayes Algorithm

ขั้นตอนหลัก:
    1. โหลดข้อมูล Adult Dataset
    2. ทำความสะอาดข้อมูล (แปลง categorical เป็น numerical)
    3. แยก Features (X) และ Labels (y)
    4. แบ่งข้อมูลเป็น Train/Test
    5. สร้างและ Train โมเดล Naive Bayes
    6. ทำนายผลและประเมินความแม่นยำ
=============================================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# =============================================================================
# ฟังก์ชันสำหรับเตรียมข้อมูล
# =============================================================================

def clean_data(dataset):
    """
    แปลงข้อมูลประเภท object (ข้อความ) ให้เป็นตัวเลข
    เพราะ Machine Learning ต้องการข้อมูลเป็นตัวเลขเท่านั้น
    
    ตัวอย่าง: "Male" -> 1, "Female" -> 0
    """
    for column in dataset.columns:
        if dataset[column].dtype == 'object':  # ถ้าข้อมูลเป็นข้อความ
            le = LabelEncoder()  # สร้างตัวแปลง
            dataset[column] = le.fit_transform(dataset[column])  # แปลงเป็นตัวเลข
    return dataset


def split_features_class(dataset, feature):
    """
    แยกข้อมูลออกเป็น 2 ส่วน:
    - features (X): ข้อมูลที่ใช้ทำนาย (ทุกคอลัมน์ยกเว้น target)
    - labels (y): ผลลัพธ์ที่ต้องการทำนาย (คอลัมน์ income)
    """
    features = dataset.drop(feature, axis=1)  # ลบคอลัมน์ target ออก เหลือแค่ features
    labels = dataset[feature].copy()  # เอาเฉพาะคอลัมน์ target
    return features, labels


# =============================================================================
# 1. โหลดข้อมูล
# =============================================================================
dataset = pd.read_csv("adult.csv")
# print(dataset.head())  # ดูข้อมูล 5 แถวแรก

# =============================================================================
# 2. ทำความสะอาดข้อมูล (แปลง object -> ตัวเลข)
# =============================================================================
dataset = clean_data(dataset)

# =============================================================================
# 3. แยก Features (X) และ Labels (y)
# =============================================================================
# features = ข้อมูลทั้งหมดยกเว้น income (อายุ, อาชีพ, การศึกษา, ฯลฯ)
# labels = คอลัมน์ income (>50K หรือ <=50K)
features, labels = split_features_class(dataset, "income")

# =============================================================================
# 4. แบ่งข้อมูลเป็น Train/Test (80% train, 20% test)
# =============================================================================
# train_test_split คืนค่า 4 ตัว:
#   - train_features: ข้อมูล features สำหรับ train (80%)
#   - test_features: ข้อมูล features สำหรับ test (20%)
#   - train_labels: ข้อมูล labels สำหรับ train (80%)
#   - test_labels: ข้อมูล labels สำหรับ test (20%)
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# =============================================================================
# 5. สร้างและ Train โมเดล Naive Bayes
# =============================================================================
model = GaussianNB()  # สร้างโมเดล Gaussian Naive Bayes
model.fit(train_features, train_labels)  # สอนโมเดลด้วยข้อมูล train

# =============================================================================
# 6. ทำนายผลและประเมินความแม่นยำ
# =============================================================================
clf_pred = model.predict(test_features)  # ทำนายผลจากข้อมูล test
accuracy = accuracy_score(test_labels, clf_pred) * 100  # คำนวณความแม่นยำ (%)

print(f"Accuracy: {accuracy:.2f}%")  # แสดงผลความแม่นยำ