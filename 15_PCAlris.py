# ==================== PCA + Naive Bayes กับ Iris Dataset ====================
# จุดประสงค์: ใช้ PCA ลดมิติข้อมูล แล้วเทรน Naive Bayes เพื่อจำแนกดอกไม้ Iris
# ขั้นตอน: 1. Load Data → 2. PCA → 3. Train/Test Split → 4. Train Model → 5. Evaluate

# ==================== 1. Import Libraries ====================
from sklearn.model_selection import train_test_split  # แบ่งข้อมูลเป็น train/test
from sklearn.naive_bayes import GaussianNB            # โมเดล Gaussian Naive Bayes
from sklearn.metrics import accuracy_score            # วัดความแม่นยำของโมเดล
import pandas as pd                                    # จัดการ DataFrame
import matplotlib.pyplot as plt                        # สร้างกราฟ
import seaborn as sb                                   # โหลด dataset และสร้างกราฟ
from sklearn.decomposition import PCA                  # ลดมิติข้อมูลด้วย PCA

# ==================== 2. Load Dataset ====================
# โหลด Iris dataset จาก seaborn (มี 150 ตัวอย่าง, 4 features, 3 species)
iris = sb.load_dataset('iris')

# แยก Features (x) และ Target (y)
x = iris.drop('species', axis=1)  # x = 4 คอลัมน์: sepal_length, sepal_width, petal_length, petal_width
y = iris['species']               # y = ชนิดดอกไม้: setosa, versicolor, virginica

# ==================== 3. PCA - ลดมิติข้อมูล ====================
# ลดจาก 4 features เหลือ 3 Principal Components
# ทำให้ข้อมูลมีขนาดเล็กลง แต่ยังเก็บข้อมูลสำคัญไว้
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)  # fit = เรียนรู้, transform = แปลงข้อมูล

# เพิ่มคอลัมน์ PCA เข้าไปใน DataFrame
# x_pca[:, 0] = ค่า PC1 ทุกแถว, x_pca[:, 1] = ค่า PC2 ทุกแถว, ...
x['pca1'] = x_pca[:, 0]
x['pca2'] = x_pca[:, 1]
x['pca3'] = x_pca[:, 2]

# ==================== 4. Train/Test Split ====================
# แบ่งข้อมูล 80% สำหรับ train, 20% สำหรับ test
# random_state=42 ทำให้ผลลัพธ์เหมือนกันทุกครั้งที่รัน
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# เลือกใช้เฉพาะคอลัมน์ PCA (ไม่ใช้ features เดิม)
x_train = x_train.loc[:, 'pca1':'pca3']  # เลือกคอลัมน์ pca1 ถึง pca3
x_test = x_test.loc[:, 'pca1':'pca3']

# ==================== 5. Train Model ====================
# ใช้ Gaussian Naive Bayes เพราะเหมาะกับข้อมูลที่เป็นตัวเลขต่อเนื่อง
model = GaussianNB()
model.fit(x_train, y_train)  # เทรนโมเดลด้วย training data

# ==================== 6. Predict & Evaluate ====================
y_pred = model.predict(x_test)  # ทำนายผลจาก test data

# คำนวณความแม่นยำ = (จำนวนที่ทายถูก / จำนวนทั้งหมด) * 100
print("Accuracy :", accuracy_score(y_test, y_pred))
# ผลลัพธ์ประมาณ 96.67% (29/30 ถูกต้อง)