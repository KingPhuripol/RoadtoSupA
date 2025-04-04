# BirdSong Recognition Project

## โจทย์และวัตถุประสงค์

โปรเจกต์นี้เป็นงานด้าน **การจำแนกเสียงนก (Bird Song Classification)** โดยมีวัตถุประสงค์เพื่อสร้างโมเดลที่สามารถระบุสกุล (genus) ของนกจากเสียงร้องของพวกมันได้ โดยใช้เทคนิคการเรียนรู้ของเครื่อง (Machine Learning) และการเรียนรู้เชิงลึก (Deep Learning)

**ข้อมูลที่ใช้:**
- ไฟล์เสียงนกในรูปแบบ FLAC
- ข้อมูลประกอบ 2 ส่วน:
  - ข้อมูลฝึก (train) ที่มีทั้งไฟล์เสียงและป้ายกำกับ (label)
  - ข้อมูลทดสอบ (test) ที่มีเฉพาะไฟล์เสียง (ต้องทำนายป้ายกำกับ)

## วิธีการดำเนินงาน

1. **เตรียมข้อมูล (Data Preparation)**
   - โหลดและแตกไฟล์ข้อมูล
   - จัดโครงสร้างข้อมูลให้เหมาะสม
   - แปลงป้ายกำกับจากชื่อสกุลนกเป็นตัวเลข

2. **การเตรียมโมเดล (Modeling)**
   - ใช้โมเดล DistilHuBERT ซึ่งเป็นโมเดลขนาดเล็กที่ได้มาจากการทำ Knowledge Distillation จากโมเดล HuBERT
   - ทำ Feature Extraction จากเสียงนก
   - ปรับขนาดข้อมูลเสียงให้เหมาะสม

3. **การฝึกโมเดล (Training)**
   - กำหนดพารามิเตอร์การฝึก
   - ฝึกโมเดลด้วยข้อมูลที่เตรียมไว้

4. **การประเมินและทำนาย (Evaluation & Prediction)**
   - ทดสอบโมเดลกับข้อมูลทดสอบ
   - ทำนายสกุลของนกจากเสียงร้อง

## รายละเอียดโค้ด

### 1. การเตรียมข้อมูล

```python
# โหลดและแตกไฟล์ข้อมูล
!unzip /content/birdsong-recognition-cmu.zip

# นำเข้าไลบรารีที่จำเป็น
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from datasets import Dataset, Audio
```

**การแมปป้ายกำกับ:**
```python
feature_map = {
    'Acrocephalus':0, 'Anthus':1, 'Columba':2, 'Corvus':3, 'Emberiza':4,
    'Motacilla':5, 'Passer':6, 'Phylloscopus':7, 'Pluvialis':8, 'Poecile':9,
    'Streptopelia':10, 'Sylvia':11, 'Tringa':12, 'Turdus':13
}
```

**โหลดข้อมูลฝึก:**
```python
train_df = pd.read_csv('/content/train.csv')
train_df['file_path'] = '/content/Data_files/train/xc'+train_df['file_id'].astype(str) + '.flac'
train_df['genus'] = train_df['genus'].map(feature_map)
```

### 2. การเตรียม Dataset

```python
# สร้าง Dataset จาก pandas DataFrame
dataset = Dataset.from_pandas(train_df)
dataset = dataset.rename_columns({"genus": "genre", "file_path":"file"})
dataset = dataset.add_column("audio", dataset["file"])
dataset = dataset.cast_column("audio", Audio())
```

### 3. การเตรียมโมเดล

```python
# โหลด Feature Extractor
from transformers import AutoFeatureExtractor
model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)

# กำหนด sampling rate
sampling_rate = feature_extractor.sampling_rate  # 16000 Hz
```

**Preprocessing Function:**
```python
max_duration = 30.0  # ความยาวสูงสุดของไฟล์เสียง (วินาที)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs
```

**Encode ข้อมูล:**
```python
dataset_encoded = dataset.map(
    preprocess_function,
    remove_columns=["audio","file_id", "file"],
    batched=True,
    batch_size=10,
    num_proc=1,
)
dataset_encoded = dataset_encoded.rename_column("genre", "label")
```

### 4. การฝึกโมเดล

```python
from transformers import AutoModelForAudioClassification

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=14  # มี 14 สกุลของนก
)
```

**กำหนดพารามิเตอร์การฝึก:**
```python
from transformers import TrainingArguments

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-birdsong",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
    report_to="none"
)
```

**เริ่มการฝึก:**
```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_encoded,
    tokenizer=feature_extractor
)

trainer.train()
```

### 5. การทำนายและส่งผลลัพธ์

**โหลดข้อมูลทดสอบ:**
```python
test_df = pd.read_csv('/content/test.csv')
test_df['file_path'] = '/content/Data_files/test/xc'+test_df['file_id'].astype(str) + '.flac'
test_df = test_df.drop(columns='genus')
```

**สร้าง Pipeline สำหรับการจำแนกเสียง:**
```python
from transformers import pipeline

pipe = pipeline("audio-classification", model="Jamvess/distilhubert-finetuned-birdsong")
```

**ทำนายผลลัพธ์:**
```python
ans_label = []
for sample in tqdm(test_dataset):
    sample=sample['audio']['array']
    ans_label.append(pipe(sample)[0]['label'])
```

**จัดรูปแบบผลลัพธ์และบันทึก:**
```python
ans_label = [label.split('_')[1] for label in ans_label]
ans_label = [int(x) for x in ans_label]

# แปลงกลับเป็นชื่อสกุลนก
fm = {
    0:'Acrocephalus', 1:'Anthus', 2:'Columba', 3:'Corvus', 4:'Emberiza',
    5:'Motacilla', 6:'Passer', 7:'Phylloscopus', 8:'Pluvialis', 9:'Poecile',
    10:'Streptopelia', 11:'Sylvia', 12:'Tringa', 13:'Turdus'
}

test_df['genus'] = ans_label
test_df['genus'] = test_df['genus'].map(fm)
test_df = test_df.drop(columns='file_path')

# บันทึกผลลัพธ์
test_df.to_csv('signal_submit_2.csv', index=False)
```

## สรุป

โปรเจกต์นี้แสดงให้เห็นถึงกระบวนการทั้งหมดในการสร้างโมเดลสำหรับจำแนกเสียงนก ตั้งแต่การเตรียมข้อมูล การเลือกและปรับโมเดล การฝึก และการประเมินผล โดยใช้โมเดล DistilHuBERT ซึ่งเป็นโมเดลที่ออกแบบมาสำหรับงานด้านเสียงโดยเฉพาะ

**จุดเด่นของวิธีการนี้:**
- ใช้โมเดลที่ผ่านการปรับแต่งมาสำหรับงานด้านเสียง
- มีการเตรียมข้อมูลอย่างเหมาะสม
- กระบวนการทั้งหมดสามารถทำซ้ำได้ (reproducible)

**ข้อควรปรับปรุง:**
- อาจเพิ่มการ augment ข้อมูลเสียงเพื่อเพิ่มความหลากหลายของข้อมูลฝึก
- ทดลองกับโมเดลอื่นๆ เพื่อเปรียบเทียบประสิทธิภาพ
- เพิ่มการประเมินผลด้วยเมตริกต่างๆ ให้ครอบคลุมมากขึ้น
