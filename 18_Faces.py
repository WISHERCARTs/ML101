# ==================== Face Recognition - จำแนกใบหน้าด้วย PCA + SVM ====================
# จุดประสงค์: สร้างโมเดลจำแนกใบหน้าคนดังจาก dataset LFW
# ขั้นตอน: 1. Load Data → 2. PCA ลดมิติ → 3. SVM จำแนก → 4. Evaluate ด้วย Confusion Matrix
#
# Flow: รูปภาพ (2914 pixels) → PCA (150 features) → SVM → ทำนายชื่อคน

# ==================== 1. Import Libraries ====================
from sklearn.datasets import fetch_lfw_people   # โหลด dataset ใบหน้าคนดัง (LFW)
import matplotlib.pyplot as plt                  # สร้างกราฟและแสดงรูป
from sklearn.decomposition import PCA            # ลดมิติข้อมูลด้วย PCA
from sklearn.svm import SVC                      # Support Vector Classifier
from sklearn.pipeline import Pipeline            # รวม PCA + SVM เป็น pipeline เดียว
from sklearn.model_selection import train_test_split   # แบ่ง train/test
from sklearn.model_selection import GridSearchCV       # หา hyperparameter ที่ดีที่สุด
from sklearn.metrics import accuracy_score, confusion_matrix  # วัดผล
import seaborn as sb                             # สร้าง heatmap สวยๆ

# ==================== 2. โหลด Dataset ====================
# LFW = Labeled Faces in the Wild (รูปใบหน้าจริงจากอินเทอร์เน็ต)
# min_faces_per_person=60 : เลือกเฉพาะคนที่มีรูปอย่างน้อย 60 รูป (8 คน)
faces = fetch_lfw_people(min_faces_per_person=60)

# โครงสร้างข้อมูล:
# faces.data.shape = (1348, 2914) → 1348 รูป, แต่ละรูปมี 2914 pixels (62x47)
# faces.target = [0, 1, 2, ...] → index บอกว่าคนนี้เป็นใคร
# faces.target_names = ['Ariel Sharon', 'Colin Powell', ...] → ชื่อจริง

# ==================== 3. สร้างโมเดล PCA + SVM ====================

# --- 3.1 PCA: ลดมิติข้อมูล ---
# n_components=150   : ลดจาก 2914 เหลือ 150 features
# svd_solver='randomized' : ใช้วิธีสุ่มเร็วขึ้น (สำหรับข้อมูลใหญ่)
# whiten=True        : ปรับ variance ให้เท่ากัน (ช่วย SVM ทำงานดีขึ้น)
pca = PCA(n_components=150, svd_solver='randomized', whiten=True)

# --- 3.2 SVM: จำแนกข้อมูล ---
# kernel='rbf'       : ใช้ Radial Basis Function (เส้นแบ่งโค้งได้)
# class_weight='balanced' : ปรับน้ำหนักให้ class ที่มีน้อยไม่โดนละเลย
svc = SVC(kernel='rbf', class_weight='balanced')

# --- 3.3 Pipeline: รวม PCA + SVM เป็นขั้นตอนเดียว ---
# ข้อดี: เรียก fit() ครั้งเดียว ทำทั้ง PCA และ SVM เลย
model = Pipeline([('pca', pca), ('svc', svc)])

# ==================== 4. แบ่งข้อมูล Train/Test ====================
# 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    faces.data, faces.target, test_size=0.2, random_state=0
)

# ==================== 5. GridSearchCV: หา Hyperparameter ที่ดีที่สุด ====================
# ลองทุกคู่ของ C และ gamma แล้วเลือกตัวที่ดีที่สุด
# svc__C     : ค่า regularization (ยิ่งมาก = ยิ่งพยายาม fit ข้อมูล train)
# svc__gamma : ค่าความกว้างของ RBF kernel (ยิ่งมาก = เส้นแบ่งยิ่งโค้ง)
param = {
    'svc__C': [1, 5, 10, 50, 100],
    'svc__gamma': [0.0001, 0.005, 0.001, 0.05]
}

# cv=5 : ใช้ 5-fold cross validation
grid = GridSearchCV(model, param_grid=param, cv=5)
grid.fit(x_train, y_train)  # ลองทุกคู่ parameter (อาจใช้เวลานาน)

# ใช้โมเดลที่ดีที่สุดจาก GridSearchCV
# grid.best_params_    = {'svc__C': 10, 'svc__gamma': 0.001} (ตัวอย่าง)
# grid.best_score_     = 0.85 (ค่า accuracy บน training set)
model = grid.best_estimator_

# ==================== 6. Predict & Evaluate ====================

# --- 6.1 ทำนาย ---
y_pred = model.predict(x_test)  # ทำนายจาก test set

# --- 6.2 วัด Accuracy ---
# Accuracy = (จำนวนที่ทายถูก / จำนวนทั้งหมด) * 100
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# --- 6.3 Confusion Matrix ---
# แสดงว่าโมเดลเดาถูก/ผิดอย่างไร
# - แนวทแยง (diagonal) = ทายถูก
# - นอกแนวทแยง = ทายผิด (เช่น เดาว่า Colin Powell แต่จริงๆ คือ George Bush)
confusion = confusion_matrix(y_test, y_pred)

# สร้าง Heatmap แสดง Confusion Matrix
sb.heatmap(
    confusion, 
    annot=True,           # แสดงตัวเลขในช่อง
    fmt='d',              # format เป็น integer
    cmap='Blues',         # สี
    cbar=False,           # ไม่แสดง colorbar
    xticklabels=faces.target_names,  # ชื่อแกน X (Predicted)
    yticklabels=faces.target_names   # ชื่อแกน Y (True)
)
plt.xlabel("Predicted Label")  # ค่าที่โมเดลทำนาย
plt.ylabel("True Label")       # ค่าจริง
plt.title("Confusion Matrix")
plt.show()

# 💡 อ่าน Confusion Matrix:
#    - ดูแนวทแยง: ยิ่งเลขสูง = โมเดลทายถูกมาก
#    - ดูนอกแนวทแยง: บอกว่าโมเดลสับสนระหว่างใครกับใคร

