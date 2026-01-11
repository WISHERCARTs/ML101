"""
ML4.py - Iris Dataset Visualization with Pair Plot
วัตถุประสงค์: สำรวจข้อมูล (EDA) ดอกไอริส 3 สายพันธุ์ 
             โดยดูความสัมพันธ์ระหว่าง features ต่างๆ
"""

# Import Libraries
import seaborn as sb           # สร้างกราฟ (built on matplotlib)
import matplotlib.pyplot as plt 

# Load Data
# Iris dataset มี 150 ตัวอย่าง, 4 features:
# - sepal_length, sepal_width (กลีบเลี้ยง)
# - petal_length, petal_width (กลีบดอก)
# - species: setosa, versicolor, virginica (3 สายพันธุ์)
iris_dataset = sb.load_dataset("iris")

# print(iris_dataset.head())  # ดู 5 แถวแรก (uncomment เพื่อ debug)

# Visualization
sb.set()  # ใช้ default style ของ seaborn (grid สวยๆ)

# Pair Plot: แสดงความสัมพันธ์ทุกคู่ของ features
# - เส้นทแยงมุม = histogram แสดงการกระจายของแต่ละ feature
# - นอกเส้นทแยง = scatter plot ดูความสัมพันธ์ระหว่าง 2 features
# - hue="species" = แยกสีตามสายพันธุ์ (3 สี = 3 ชนิด)
# - height=2 = ขนาดของแต่ละกราฟย่อย (นิ้ว)
# TIP: petal_length vs petal_width แยก species ได้ชัดที่สุด!
sb.pairplot(iris_dataset, hue="species", height=2) 

plt.show()