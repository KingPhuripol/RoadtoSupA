# Heart Disease Prediction using AutoGluon

## โจทย์ปัญหา

โจทย์นี้เป็นปัญหาการทำนายโรคหัวใจ (Heart Disease Prediction) โดยใช้ข้อมูลจากชุดข้อมูลสุขภาพต่างๆ ของผู้ป่วย เช่น ความดันโลหิตสูง คอเลสเตอรอล ดัชนีมวลกาย ประวัติการสูบบุหรี่ ฯลฯ เพื่อทำนายว่าผู้ป่วยมีประวัติโรคหัวใจหรือหัวใจวายหรือไม่ ("Yes" หรือ "No")

## วิธีการแก้ปัญหา

เราจะใช้ AutoGluon ซึ่งเป็น AutoML framework ที่สามารถสร้างโมเดล Machine Learning ได้โดยอัตโนมัติ โดยมีขั้นตอนหลักๆ ดังนี้:

1. **เตรียมข้อมูล**:
   - ลบข้อมูลที่ขาดหายไปในคอลัมน์เป้าหมาย
   - สร้างสมดุลของคลาส (class balance) เนื่องจากข้อมูลมีคลาสไม่สมดุล
   - สร้างฟีเจอร์ใหม่จากข้อมูลที่มีอยู่

2. **สร้างและฝึกโมเดล**:
   - ใช้ AutoGluon's TabularPredictor
   - กำหนด hyperparameters สำหรับอัลกอริธึมต่างๆ
   - ฝึกโมเดลด้วยเวลา limit 10 นาที

3. **ทำนายผลและส่ง submission**:
   - โหลดข้อมูลทดสอบ
   - ทำนายผลและบันทึกผลลัพธ์

## รายละเอียดโค้ด

### 1. ติดตั้งและนำเข้าไลบรารี

```python
pip -q install autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
```

### 2. โหลดข้อมูล

```python
train_data = pd.read_csv("/kaggle/input/hearth-disease-recognition/train.csv")
```

### 3. ตรวจสอบและเตรียมข้อมูล

```python
# ตรวจสอบการกระจายของคลาส
train_data['History of HeartDisease or Attack'].value_counts()
# Output: No (203322), Yes (18068) - คลาสไม่สมดุล

# ลบแถวที่ข้อมูลเป้าหมายหายไป
train_data = train_data.dropna(subset=['History of HeartDisease or Attack'])

# สร้างสมดุลคลาสโดยสุ่มตัวอย่าง
df_yes = train_data[train_data['History of HeartDisease or Attack'] == 'Yes'].sample(n=18068, random_state=42)
df_no = train_data[train_data['History of HeartDisease or Attack'] == 'No'].sample(n=20000, random_state=42)
train_data = pd.concat([df_yes, df_no], ignore_index=True)
```

### 4. สร้างฟีเจอร์ใหม่

```python
def create_new_features(train_data):
    # แปลงค่า Yes/No เป็น 1/0
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = [
        "High Blood Pressure", "Told High Cholesterol", "Cholesterol Checked",
        "Smoked 100+ Cigarettes", "Diagnosed Stroke", "Diagnosed Diabetes",
        "Leisure Physical Activity", "Heavy Alcohol Consumption",
        "Health Care Coverage", "Doctor Visit Cost Barrier",
        "Difficulty Walking", "Vegetable or Fruit Intake (1+ per Day)"
    ]
    
    for col in binary_cols:
        if train_data[col].dtype == object:
            train_data[col + "_bin"] = train_data[col].map(binary_map)
        else:
            train_data[col + "_bin"] = train_data[col]

    # สร้างฟีเจอร์ BMI Category
    def categorize_bmi(bmi):
        if bmi < 18.5: return "Underweight"
        elif bmi < 25: return "Normal weight"
        elif bmi < 30: return "Overweight"
        else: return "Obese"
    
    train_data['BMI Category'] = train_data['Body Mass Index'].apply(categorize_bmi)

    # สร้างฟีเจอร์ Obesity Risk
    train_data['Obesity Risk'] = ((train_data['Body Mass Index'] >= 30) & 
                                 (train_data['Leisure Physical Activity'] == 'No')).astype(int)

    # ลบคอลัมน์ที่ไม่ต้องการ
    columns_to_drop = [...]
    train_data = train_data.drop(columns=columns_to_drop)
    
    return train_data
```

### 5. ฝึกโมเดลด้วย AutoGluon

```python
save_path = 'best_model'
hyperparameters = {
    'GBM': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
    'CAT': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
    'XGB': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
    'RF': [{'ag_args_fit': {'num_gpus': 0}}, {'ag_args_fit': {'num_gpus': 1}}],
}
time_limit=600  # 10 นาที

predictor = TabularPredictor(
    label='History of HeartDisease or Attack',
    problem_type='binary',
    path=save_path
).fit(
    train_data,
    presets='best_quality',
    hyperparameters=hyperparameters,
    time_limit=time_limit
)
```

### 6. ทำนายผลและส่ง submission

```python
# โหลดข้อมูลทดสอบ
test_data = pd.read_csv("/kaggle/input/hearth-disease-recognition/test.csv")

# เตรียมข้อมูลทดสอบ
test_data = create_new_features(test_data)

# ทำนายผล
y_pred = predictor.predict_proba(test_data)

# กำหนด threshold ที่ 0.505 สำหรับการแบ่งคลาส
y_pred['outcome'] = y_pred.apply(lambda row: 'Yes' if row['Yes'] > 0.505 else 'No', axis=1)

# เตรียมไฟล์ส่ง
sample_submission = pd.read_csv("/kaggle/input/hearth-disease-recognition/sample_submission.csv")
sample_submission['History of HeartDisease or Attack'] = y_pred['outcome']
sample_submission.to_csv("submission.csv", index=False)
```

## สรุป

- เราใช้ AutoGluon เพื่อสร้างโมเดลทำนายโรคหัวใจโดยอัตโนมัติ
- มีการปรับสมดุลคลาสและสร้างฟีเจอร์ใหม่เพื่อเพิ่มประสิทธิภาพ
- โมเดลที่ดีที่สุดได้ accuracy ประมาณ 78.9% บน validation set
- ไฟล์ submission สุดท้ายมีประมาณ 50,155 "No" และ 24,206 "Yes"
