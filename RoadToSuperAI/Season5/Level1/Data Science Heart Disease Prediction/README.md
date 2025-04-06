# 💓 Heart Disease Prediction using AutoGluon

## 📌 โจทย์ปัญหา

โครงการนี้มุ่งเน้นการ **ทำนายโรคหัวใจ** (Heart Disease Prediction) โดยใช้ข้อมูลด้านสุขภาพของผู้ป่วย เช่น:

- ความดันโลหิตสูง
- คอเลสเตอรอล
- ดัชนีมวลกาย (BMI)
- ประวัติการสูบบุหรี่  
ฯลฯ

เพื่อทำนายว่าผู้ป่วยมี **ประวัติโรคหัวใจหรือหัวใจวาย** หรือไม่ (`"Yes"` หรือ `"No"`)

---

## 🧠 วิธีการแก้ปัญหา

เราเลือกใช้ **[AutoGluon](https://auto.gluon.ai/)** ซึ่งเป็น AutoML ที่สามารถสร้างและฝึกโมเดลได้อย่างอัตโนมัติ โดยมีขั้นตอนหลักดังนี้:

### 1. เตรียมข้อมูล
- ลบแถวที่มีค่าขาดหาย (missing value) ใน target
- สร้างสมดุลของคลาส (`class balance`)
- สร้างฟีเจอร์ใหม่จากข้อมูลเดิม

### 2. สร้างและฝึกโมเดล
- ใช้ `TabularPredictor` ของ AutoGluon
- กำหนด hyperparameters
- ฝึกโมเดลภายใน 10 นาที

### 3. ทำนายผลและส่ง submission
- โหลดชุดข้อมูลทดสอบ
- ทำนายและแปลงผลเป็น `"Yes"` หรือ `"No"`
- สร้างไฟล์ `submission.csv`

---

## 💻 รายละเอียดโค้ด

### 🔧 ติดตั้งและนำเข้าไลบรารี

```python
pip -q install autogluon

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
```

---

### 📥 โหลดข้อมูล

```python
train_data = pd.read_csv("/kaggle/input/hearth-disease-recognition/train.csv")
```

---

### 🔎 ตรวจสอบและเตรียมข้อมูล

```python
# ตรวจสอบ class distribution
train_data['History of HeartDisease or Attack'].value_counts()
# Output: No (203,322), Yes (18,068)

# ลบแถวที่มี missing target
train_data = train_data.dropna(subset=['History of HeartDisease or Attack'])

# สุ่มตัวอย่างให้สมดุล
df_yes = train_data[train_data['History of HeartDisease or Attack'] == 'Yes'].sample(n=18068, random_state=42)
df_no = train_data[train_data['History of HeartDisease or Attack'] == 'No'].sample(n=20000, random_state=42)
train_data = pd.concat([df_yes, df_no], ignore_index=True)
```

---

### 🧱 สร้างฟีเจอร์ใหม่

```python
def create_new_features(train_data):
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = [
        "High Blood Pressure", "Told High Cholesterol", "Cholesterol Checked",
        "Smoked 100+ Cigarettes", "Diagnosed Stroke", "Diagnosed Diabetes",
        "Leisure Physical Activity", "Heavy Alcohol Consumption",
        "Health Care Coverage", "Doctor Visit Cost Barrier",
        "Difficulty Walking", "Vegetable or Fruit Intake (1+ per Day)"
    ]
    
    # แปลงค่าทางเลือกเป็นตัวเลข
    for col in binary_cols:
        if train_data[col].dtype == object:
            train_data[col + "_bin"] = train_data[col].map(binary_map)
        else:
            train_data[col + "_bin"] = train_data[col]

    # ฟีเจอร์หมวดหมู่ BMI
    def categorize_bmi(bmi):
        if bmi < 18.5: return "Underweight"
        elif bmi < 25: return "Normal weight"
        elif bmi < 30: return "Overweight"
        else: return "Obese"
    
    train_data['BMI Category'] = train_data['Body Mass Index'].apply(categorize_bmi)

    # ฟีเจอร์ Obesity Risk
    train_data['Obesity Risk'] = ((train_data['Body Mass Index'] >= 30) & 
                                 (train_data['Leisure Physical Activity'] == 'No')).astype(int)

    # ลบคอลัมน์ที่ไม่จำเป็น (ระบุคอลัมน์ที่ต้องการลบแทน [...])
    columns_to_drop = [...]  
    train_data = train_data.drop(columns=columns_to_drop)
    
    return train_data
```

---

### 🤖 ฝึกโมเดลด้วย AutoGluon

```python
save_path = 'best_model'

hyperparameters = {
    'GBM': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
    'CAT': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
    'XGB': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
    'RF':  [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
}

predictor = TabularPredictor(
    label='History of HeartDisease or Attack',
    problem_type='binary',
    path=save_path
).fit(
    train_data,
    presets='best_quality',
    hyperparameters=hyperparameters,
    time_limit=600  # จำกัดเวลา 10 นาที
)
```

---

### 📊 ทำนายผลและสร้าง submission

```python
# โหลดข้อมูลทดสอบ
test_data = pd.read_csv("/kaggle/input/hearth-disease-recognition/test.csv")
test_data = create_new_features(test_data)

# ทำนายโอกาส
y_pred = predictor.predict_proba(test_data)

# กำหนด threshold ที่ 0.505
y_pred['outcome'] = y_pred.apply(lambda row: 'Yes' if row['Yes'] > 0.505 else 'No', axis=1)

# เตรียมไฟล์ submission
sample_submission = pd.read_csv("/kaggle/input/hearth-disease-recognition/sample_submission.csv")
sample_submission['History of HeartDisease or Attack'] = y_pred['outcome']
sample_submission.to_csv("submission.csv", index=False)
```

---

## ✅ สรุป

- ใช้ **AutoGluon** สำหรับสร้างโมเดลโดยอัตโนมัติ
- ปรับสมดุลข้อมูลและสร้างฟีเจอร์ใหม่เพื่อเพิ่มประสิทธิภาพ
- Accuracy บน validation set ≈ **78.9%**
- ไฟล์ submission สุดท้ายมี:
  - `"No"` ≈ **50,155 คน**
  - `"Yes"` ≈ **24,206 คน**

