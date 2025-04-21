## 🔍 **ส่วนที่ 1: เตรียมและประมวลผลไฟล์ PDF**

### 📦 โหลดไลบรารี
```python
import fitz  # หรือ PyMuPDF สำหรับอ่าน PDF
import json, os
from tqdm.auto import tqdm, trange

# สำหรับจัดการเอกสาร
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
```
- `fitz` ใช้อ่านไฟล์ PDF ทีละหน้า
- `tqdm` แสดง progress bar
- `CharacterTextSplitter` แบ่งข้อความเป็นชิ้นย่อย ๆ (chunk) สำหรับฝัง (embedding)

---

### 🔧 ฟังก์ชันช่วยจัดรูปแบบ
```python
def process_text(text):
    return text.replace(' า','ำ')

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["file"] = record["file"]
    metadata["page"] = record["page"]
    metadata['source'] = '-'
    return metadata
```

- `process_text` แก้ปัญหา encoding ภาษาไทย (แก้ `' า'` ให้เป็น `'ำ'`)
- `metadata_func` ใส่ข้อมูลประกอบในแต่ละหน้า เช่น หมายเลขหน้า, ชื่อไฟล์ ฯลฯ

---

### 📄 แปลง PDF เป็น JSON
```python
pdf_file = '/content/สรุปผลงานสภาผู้แทนราษฎร_ชุดที่_26_ปีที่_1_ครั้งที่หนึ่ง.pdf'
document = fitz.open(pdf_file)
data_json = []

for page_num in range(len(document)):
    page = document.load_page(page_num)
    text = page.get_text("text")
    data_json.append({
        'page': page_num,
        'text': process_text(text),
        'file': pdf_file[1:]
    })

with open("temp_pdf.json", "w", encoding="utf-8") as json_file:
    json.dump({'pdf': data_json}, json_file, ensure_ascii=False, indent=4)
```

- เปิด PDF และดึงข้อความออกจากแต่ละหน้า
- แปลงข้อมูลให้เก็บในรูปแบบ JSON พร้อม metadata

---

## 📂 **ส่วนที่ 2: โหลด JSON เข้า LangChain และแบ่งข้อความ**

```python
loader = JSONLoader(
    file_path=output_file_path,
    jq_schema='."pdf"[]',
    content_key="text",
    metadata_func=metadata_func,
    text_content=False
)
pages = loader.load()
```

- ใช้ `JSONLoader` เพื่อโหลดไฟล์ JSON ที่เราแปลงไว้
- ใช้ `jq_schema` เพื่อระบุว่าจะดึงข้อมูลจาก key `"pdf"` (ซึ่งเป็น list ของหน้า)
- `content_key="text"` บอกว่าเนื้อหาหลักอยู่ใน field `"text"`

---

### ✂️ แบ่งข้อความออกเป็นชิ้นย่อย
```python
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1500,
    chunk_overlap=0,
    separator="\n\n"
)
docs = text_splitter.split_documents(pages)
```

- แบ่งเอกสารที่โหลดออกมาเป็น **chunk ยาว 1500 token** โดยไม่มี overlap
- การแยกนี้ช่วยให้เหมาะกับ LLM ที่มี context window จำกัด และเหมาะสำหรับการฝังข้อมูล (embedding)

---

### 🧹 ลบไฟล์ชั่วคราว
```python
if os.path.exists(output_file_path):
    os.remove(output_file_path)
```

- ลบไฟล์ `temp_pdf.json` ที่สร้างไว้แล้วหลังโหลดเข้าโปรแกรมเรียบร้อย

---

## 🧠 **ส่วนที่ 3: เตรียมระบบฐานข้อมูลเวกเตอร์ (Qdrant)**

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

url = "https://1d11517d-916b-48b0-b59f-8e9083ab4a37.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key = userdata.get('QDRANT_TOKEN')

client = QdrantClient(url=url, api_key=api_key)
```

- เชื่อมต่อกับ **Qdrant Cloud** ซึ่งเป็นฐานข้อมูลเวกเตอร์แบบ vector store (ใช้สำหรับ similarity search)
- ใช้ API Key ที่เก็บไว้ในตัวแปร `userdata`

---
