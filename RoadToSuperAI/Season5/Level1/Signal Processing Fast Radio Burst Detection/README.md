## 🌌 อธิบายโจทย์: ตรวจจับสัญญาณ Fast Radio Burst (FRB)

โค้ดนี้เขียนด้วยภาษา **Python** ในรูปแบบของ **Jupyter Notebook** โดยมีวัตถุประสงค์เพื่อ **วิเคราะห์และตรวจจับสัญญาณ FRB** (Fast Radio Burst) จากข้อมูลที่ให้มาในรูปแบบ `.csv` และ `.npy` ซึ่งประกอบด้วย **สัญญาณคลื่นวิทยุ** และ **ป้ายกำกับ (labels)**

---

## 🧭 ขั้นตอนการทำงาน

### 1. 🔍 การสำรวจข้อมูลเบื้องต้น (EDA)
- ค้นหาและรวบรวมป้ายกำกับทั้งหมดจากไฟล์ `.csv` ภายในโฟลเดอร์ที่กำหนด
- กำจัดป้ายกำกับที่ไม่ชัดเจน เช่น `'Unlabeled'` และ `'Uncertain'`

### 2. 📂 การโหลดข้อมูล
- โหลดไฟล์ `.npy` สำหรับข้อมูลสัญญาณ
- โหลดไฟล์ `.csv` สำหรับข้อมูลป้ายกำกับ
- ใช้ไลบรารี `numpy`, `pandas`, และ `glob` เพื่อจัดการข้อมูล

### 3. 🧪 การตรวจสอบตัวอย่างข้อมูล
- แสดงข้อมูลของตัวอย่างแรก
- ตรวจสอบป้ายกำกับด้วย `value_counts()` เพื่อดูการกระจายของข้อมูล

### 4. 📊 การแสดงผลข้อมูลด้วยกราฟ
- ใช้ `matplotlib` สร้างกราฟเพื่อแสดงข้อมูลสัญญาณในรูปแบบ:
  - **Dynamic Spectrum**: ภาพรวมความเข้มของสัญญาณในแต่ละช่องความถี่และเวลา
  - **Power Spectrum**: ความเข้มเฉลี่ยในแต่ละช่องความถี่
  - **Light Curve**: ความเข้มรวมตามเวลา

### 5. 🧠 การเตรียมข้อมูลสำหรับโมเดล Machine Learning
- ใช้ `Dataset` และ `DataLoader` จาก PyTorch เพื่อเตรียมข้อมูลสำหรับการฝึกโมเดลในอนาคต

---

## 🔎 อธิบายโค้ดทีละส่วน

### 📁 1. การสำรวจป้ายกำกับ
```python
import os
import pandas as pd

csv_directory = "/kaggle/input/signal-fast-radio-burst-detection/train-labels-corrected/train"
unique_labels = set()

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(csv_path)
        if "labels" in df.columns:
            unique_labels.update(df["labels"].unique())

print("Unique Labels Found:")
print(unique_labels)
```

---

### 🧾 2. การโหลดข้อมูล
```python
import numpy as np
import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader

dataset_dir = "/kaggle/input/signal-fast-radio-burst-detection/train/train"
dataset_label_dir = "/kaggle/input/signal-fast-radio-burst-detection/train-labels-corrected/train"

X_Training_Files = sorted(glob.glob(os.path.join(dataset_dir, "*.npy")))
Y_Training_Files = sorted(glob.glob(os.path.join(dataset_label_dir, "*.csv")))
```

---

### 🔬 3. ตรวจสอบข้อมูลตัวอย่าง
```python
rec_num = 0
X_demo = np.load(X_Training_Files[rec_num])
Y_demo = pd.read_csv(Y_Training_Files[rec_num])
Y_demo["labels"].value_counts()
```

---

### 🖼️ 4. ฟังก์ชันแสดงผลสัญญาณ
```python
import matplotlib.pyplot as plt

def make_plot(X_demo, title_string=None, peak=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                           left=0.1, right=0.9, bottom=0.1, top=0.93,
                           wspace=0, hspace=0)

    dynanic_spectrum = np.transpose(X_demo)

    ax1 = fig.add_subplot(gs[1, 0])
    heatmap = ax1.imshow(dynanic_spectrum, aspect="auto", origin='upper',
                         interpolation="none", cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Channel")
    ax1.set_ylim(dynanic_spectrum.shape[0]-1, 0)

    if peak is not False:
        if peak is None:
            peak = np.argmax(np.sum(dynanic_spectrum, axis=0))
        else:
            peak = peak * 256
        offset = 256
        ax1.set_xlim(peak, peak+offset)

    ax2 = fig.add_subplot(gs[1, 1])
    powerspectrum = np.sum(dynanic_spectrum, axis=1)
    freq_bin = np.linspace(0, len(powerspectrum), len(powerspectrum))
    ax2.plot(powerspectrum, freq_bin)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_xlim(0, 1.2 * np.max(powerspectrum))
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel("SED")

    ax3 = fig.add_subplot(gs[0, 0])
    lightcurve = np.sum(dynanic_spectrum, axis=0)
    ax3.plot(lightcurve)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_ylabel(r"Intensity")
    if title_string is not None:
        ax3.set_title(title_string)

    fig.colorbar(heatmap, ax=ax2, orientation="vertical", fraction=0.5)
    plt.show()

make_plot(X_demo, title_string="Dynamic Spectrum (Not Normalized, Broad+Pulse)", peak=908)
```
