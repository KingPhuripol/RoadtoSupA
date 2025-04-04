# 🌟 **การจำแนกความสุกของสับปะรดด้วย Vision Transformer (ViT)** 🌟

## 📌 **วัตถุประสงค์โครงการ**
✨ สร้างระบบอัจฉริยะสำหรับจำแนกระดับความสุกของสับปะรดจากภาพถ่าย  
✨ ใช้เทคนิค Deep Learning แบบ Fine-tuning โมเดล ViT  
✨ พัฒนาโมเดลที่มีความแม่นยำสูงเพื่อประยุกต์ใช้ในอุตสาหกรรมการเกษตร  

---

## 🛠️ **ขั้นตอนการทำงาน**

### 1. **เตรียมข้อมูล (Data Preparation)** 📂
```python
from datasets import load_dataset
dataset = load_dataset('imagefolder', data_dir='/content/train/train', split='train')
```
- โหลดชุดข้อมูลภาพสับปะรดจากโฟลเดอร์
- ข้อมูลถูกแบ่งอัตโนมัติตามโฟลเดอร์ย่อย (แต่ละโฟลเดอร์แทนระดับความสุกต่างกัน)

---

### 2. **ปรับปรุงข้อมูล (Data Preprocessing)** 🖼️
```python
from transformers import AutoImageProcessor

# ใช้ ViT Processor สำหรับปรับภาพให้เหมาะสมกับโมเดล
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(example_batch):
    images = [x.convert("RGB") for x in example_batch['image']]  # แปลงเป็น RGB
    inputs = image_processor(images, return_tensors='pt')  # ปรับขนาดและแปลงเป็น Tensor
    inputs['labels'] = example_batch['label']  # เพิ่ม Label
    return inputs
```
- ปรับขนาดภาพเป็น 224x224 พิกเซล (ตามข้อกำหนดของ ViT)
- แปลงภาพให้เป็น Tensor สำหรับประมวลผลโดยโมเดล

---

### 3. **สร้างและฝึกโมเดล (Model Training)** 🧠
```python
from transformers import AutoModelForImageClassification

# โหลดโมเดล ViT พร้อมปรับ Head สำหรับงาน Classification
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},  # Mapping ID -> Label
    label2id={c: str(i) for i, c in enumerate(labels)}   # Mapping Label -> ID
)
```

#### ⚙️ **การตั้งค่าการฝึก (Training Configuration)**
```python
training_args = TrainingArguments(
    output_dir="./vit-pineapple",
    per_device_train_batch_size=8,    # Batch Size = 8
    num_train_epochs=4,               # ฝึก 4 รอบ
    learning_rate=2e-4,               # อัตราการเรียนรู้
    evaluation_strategy="steps",      # ประเมินทุก 100 step
    save_steps=100,                   # บันทึกโมเดลทุก 100 step
)
```

#### 📊 **ผลการฝึก (Training Results)**
| Step | Training Loss | Validation Loss | Accuracy |
|------|---------------|------------------|----------|
| 100  | 0.3938        | 0.4093           | 85.41%   |
| 200  | 0.3457        | 0.4389           | 85.13%   |
| ...  | ...           | ...              | ...      |
| 1000 | 0.0629        | 0.0516           | 98.88%   |

---

### 4. **ประเมินและบันทึกโมเดล (Evaluation & Saving)** 💾
```python
# ประเมินประสิทธิภาพ
trainer.evaluate()

# บันทึกโมเดล
trainer.save_model("/content/drive/MyDrive/vit_pineapple_model")
```

---

### 5. **ทำนายภาพใหม่ (Prediction)** 🔮
```python
# โหลดโมเดลที่ฝึกแล้ว
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# ทำนายภาพตัวอย่าง
image = Image.open("pineapple.jpg").convert("RGB")
inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
```

---

## 🎯 **ผลลัพธ์และการประยุกต์ใช้**
- 🚀 ความแม่นยำสูงสุด **98.88%** ในการจำแนกระดับความสุก
- 🏭 สามารถประยุกต์ใช้ในสายการผลิตเพื่อจัดเรียงสับปะรดอัตโนมัติ
- 👨‍🌾 ช่วยเกษตรกรตัดสินใจเวลาเก็บเกี่ยวที่เหมาะสม

---

## 📂 **โครงสร้างไฟล์**
```
/pineapple-project
│
├── /train              # ข้อมูลฝึกอบรม
│   ├── /unripe         # สับปะรดดิบ
│   ├── /half-ripe      # สับปะรดเกือบสุก
│   └── /ripe           # สับปะรดสุก
│
├── /test               # ข้อมูลทดสอบ
├── model.ipynb         # Notebook การฝึกโมเดล
└── requirements.txt    # ไลบรารีที่จำเป็น
```

---

## 📝 **สรุป**
โครงการนี้แสดงให้เห็นถึงประสิทธิภาพของ **Vision Transformer** ในการแก้ปัญหาการจำแนกภาพทางการเกษตร โดยได้พัฒนาระบบที่สามารถระบุระดับความสุกของสับปะรดได้อย่างแม่นยำ ซึ่งสามารถขยายผลไปสู่สินค้าเกษตรอื่นๆ ในอนาคต
