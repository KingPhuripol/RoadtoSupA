
## 📌 Image Processing Thai Language Image Captioning

โค้ดนี้ถูกออกแบบมาเพื่อ **สร้างระบบที่สามารถดูภาพและอธิบายชื่อสถานที่ในภาพเป็นภาษาไทย**  
โดยใช้โมเดล AI ที่ชื่อว่า **Typhoon2-Qwen2VL-7B-Vision-Instruct** ซึ่งมีความสามารถในการ **เข้าใจภาพพร้อมกับคำถามจากมนุษย์ แล้วตอบกลับเป็นข้อความภาษาไทยได้**

ตัวอย่าง:  
หากให้โมเดลดูภาพพระบรมมหาราชวัง โมเดลจะตอบว่า:

```
['พระบรมมหาราชวัง, กรุงเทพฯ, ประเทศไทย']
```

---

## 🧠 โมเดลที่ใช้คืออะไร?

> `Typhoon2-Qwen2VL-7B-Vision-Instruct`  
เป็นโมเดล AI ขนาดใหญ่ที่:
- มองเห็นภาพ (Vision)
- เข้าใจคำถามของมนุษย์ (Language)
- ตอบคำถามเกี่ยวกับภาพได้ (Image Captioning / Visual QA)

---

## 🧩 อธิบายแต่ละส่วนของโค้ด

---

### 🔧 1. **ติดตั้งและโหลดเครื่องมือที่จำเป็น**

```python
from PIL import Image
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
```

📌 ใช้สำหรับ:
- เปิดภาพ (`PIL`)
- ดาวน์โหลดภาพจาก URL (`requests`)
- ใช้โมเดล AI (`transformers` จาก Hugging Face)

---

### 🧠 2. **โหลดโมเดลและตัวประมวลผล**

```python
model_name = "scb10x/typhoon2-qwen2vl-7b-vision-instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
```

📌 ใช้สำหรับ:
- โหลดโมเดล Typhoon2 จาก Hugging Face
- โหลดตัวประมวลผล (processor) ที่ใช้แปลงภาพและคำถามให้โมเดลเข้าใจได้

---

### 🖼️ 3. **ดาวน์โหลดภาพจากอินเทอร์เน็ต**

```python
url = "https://cdn.pixabay.com/photo/2023/05/16/09/15/bangkok-7997046_1280.jpg"
image = Image.open(requests.get(url, stream=True).raw)
```

📌 ใช้สำหรับ:
- โหลดภาพจากเว็บไซต์ Pixabay
- แปลงให้เป็นวัตถุภาพที่โมเดลสามารถใช้งานได้

---

### 💬 4. **ตั้งคำถามให้โมเดล**

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "ระบุชื่อสถานที่และประเทศของภาพนี้เป็นภาษาไทย"},
        ],
    }
]
```

📌 ใช้สำหรับ:
- ตั้งคำถามว่า **"ภาพนี้คือที่ไหน?"**
- โมเดลจะตอบชื่อสถานที่และประเทศเป็นภาษาไทย

---

### 🛠️ 5. **เตรียมข้อมูลสำหรับโมเดล**

```python
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")
```

📌 ใช้สำหรับ:
- แปลง `conversation` ให้เป็นรูปแบบที่โมเดลเข้าใจ
- รวมภาพและข้อความเข้าด้วยกัน
- ส่งข้อมูลไปยัง GPU เพื่อประมวลผลเร็วขึ้น

---

### 🔮 6. **ให้โมเดลสร้างคำตอบ**

```python
output_ids = model.generate(**inputs, max_new_tokens=128)
```

📌 ใช้สำหรับ:
- สร้างคำตอบใหม่จากโมเดล โดยให้มีความยาวไม่เกิน 128 tokens

---

### ✂️ 7. **ตัดเอาเฉพาะคำตอบ**

```python
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
```

📌 ใช้สำหรับ:
- ตัดเอาเฉพาะคำตอบที่โมเดลสร้าง ไม่รวมคำถามเดิม

---

### 📝 8. **แปลงรหัสคำตอบเป็นข้อความ**

```python
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)
```

📌 ใช้สำหรับ:
- แปลงคำตอบให้อยู่ในรูปแบบข้อความ
- แสดงผลลัพธ์ เช่น: `['พระบรมมหาราชวัง, กรุงเทพฯ, ประเทศไทย']`

---

### 📦 9. **โหลดชุดข้อมูลเพิ่มเติม (กรณีใช้กับ Kaggle)**

```python
!kaggle competitions download -c image-processing-thai-language-image-captioning
!unzip image-processing-thai-language-image-captioning.zip
```
