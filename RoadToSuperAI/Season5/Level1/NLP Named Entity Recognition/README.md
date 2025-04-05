# 🚀 **Named Entity Recognition - NER** 🚀

### 1. **ติดตั้งไลบรารีที่จำเป็น** 📦
```python
%pip install -q datasets transformers[sentencepiece] simpletransformers scikit-learn
```
- **ไลบรารีที่ติดตั้ง:**
  - 🏗️ `datasets`: สำหรับโหลดและจัดการชุดข้อมูล
  - 🤗 `transformers`: ไลบรารีของ Hugging Face สำหรับโมเดล NLP
  - ✨ `simpletransformers`: ทำให้ใช้งานโมเดล Transformer ได้ง่ายขึ้น
  - 📊 `scikit-learn`: สำหรับการประเมินผลโมเดล

---

### 2. **เตรียมชุดข้อมูล** 📂
```python
import os
import zipfile

# กำหนดเส้นทาง
zip_path = "super-ai-ss-5-named-entity-recognition.zip"
extract_path = "super-ai-ss-5-named-entity-recognition"

# ถ้ายังไม่ได้แตกไฟล์ ให้ทำการแตกไฟล์
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # จัดโครงสร้างโฟลเดอร์ใหม่ให้เรียบง่าย
    for split in ['train', 'eval', 'test']:
        split_path = os.path.join(extract_path, split)
        nested = os.path.join(split_path, split)
        if os.path.exists(nested):
            for fname in os.listdir(nested):
                os.rename(os.path.join(nested, fname), os.path.join(split_path, fname))
            os.rmdir(nested)
```
- **สิ่งที่โค้ดทำ:**
  - 📦 แตกไฟล์ zip ที่มีชุดข้อมูล
  - 🗂️ จัดโครงสร้างโฟลเดอร์ใหม่ให้เรียบง่าย
  - **ชุดข้อมูลแบ่งเป็น 3 ส่วน:**
    - 🏋️‍♂️ `train`: สำหรับฝึกโมเดล
    - 📝 `eval`: สำหรับประเมินระหว่างฝึก
    - 🧪 `test`: สำหรับทดสอบสุดท้าย

---

### 3. **โหลดชุดข้อมูล** 🔍
```python
from datasets import load_dataset

data_files = {
    "train": "super-ai-ss-5-named-entity-recognition/train/train.csv",
    "validation": "super-ai-ss-5-named-entity-recognition/eval/eval.csv",
    "test": "super-ai-ss-5-named-entity-recognition/test/test.csv"
}

dataset = load_dataset("csv", data_files=data_files)
dataset
```
- **สิ่งที่โค้ดทำ:**
  - 📥 ใช้ไลบรารี `datasets` เพื่อโหลดข้อมูลจากไฟล์ CSV ทั้งสามส่วน
  - 📊 แสดงข้อมูลเกี่ยวกับชุดข้อมูลที่โหลดมา

---

### 4. **เตรียมและฝึกโมเดล** 🧠
```python
from simpletransformers.ner import NERModel, NERArgs
import pandas as pd

# เตรียมข้อมูลสำหรับฝึกและประเมิน
train_df = pd.read_csv("super-ai-ss-5-named-entity-recognition/train/train.csv")
eval_df = pd.read_csv("super-ai-ss-5-named-entity-recognition/eval/eval.csv")

# กำหนดค่าต่างๆ สำหรับโมเดล
model_args = NERArgs()
model_args.num_train_epochs = 3  # ฝึก 3 รอบ
model_args.train_batch_size = 8  # ขนาดชุดข้อมูลสำหรับฝึกแต่ละครั้ง
model_args.evaluate_during_training = True  # ประเมินผลระหว่างฝึก
model_args.labels_list = list(train_df['labels'].unique())  # หมวดหมู่ของเอนทิตี

# สร้างโมเดล NER
model = NERModel(
    "bert",  # ประเภทของโมเดล
    "bert-base-multilingual-cased",  # ใช้โมเดล BERT ที่รองรับหลายภาษา
    args=model_args,
    use_cuda=False  # ไม่ใช้ GPU
)

# ฝึกโมเดล
model.train_model(train_df, eval_data=eval_df)
```
- **สิ่งที่โค้ดทำ:**
  - 🛠️ ใช้ `simpletransformers` เพื่อสร้างโมเดล NER แบบง่าย
  - 🌍 เลือกใช้โมเดล **BERT รุ่นพื้นฐานที่รองรับหลายภาษา (multilingual)**
  - ⚙️ กำหนดพารามิเตอร์ต่างๆ เช่น:
    - 🔄 จำนวนรอบการฝึก (`num_train_epochs`)
    - 📦 ขนาดชุดข้อมูล (`train_batch_size`)
    - 📊 การประเมินระหว่างฝึก (`evaluate_during_training`)
  - 🏋️‍♂️ เริ่มกระบวนการฝึกโมเดลด้วยข้อมูล training และประเมินด้วยข้อมูล evaluation

---

### 5. **ประเมินโมเดลด้วยข้อมูลทดสอบ** 📊
```python
# ประเมินโมเดลด้วยข้อมูลทดสอบ
test_df = pd.read_csv("super-ai-ss-5-named-entity-recognition/test/test.csv")
result, model_outputs, predictions = model.eval_model(test_df)
print(result)
```
- **สิ่งที่โค้ดทำ:**
  - 📥 โหลดข้อมูลทดสอบ (test set)
  - 📈 ประเมินประสิทธิภาพของโมเดลที่ฝึกมาแล้ว
  - 📢 แสดงผลลัพธ์การประเมิน เช่น:
    - 🎯 ความแม่นยำ (accuracy)
    - 🔍 F1-score
    - 📌 Precision และ Recall

---
