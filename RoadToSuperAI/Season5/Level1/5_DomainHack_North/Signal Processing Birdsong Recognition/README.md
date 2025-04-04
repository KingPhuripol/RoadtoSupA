# 🐦 **BirdSong Recognition Project** 🎵

## 📌 **Project Overview**
**Objective:** Build a deep learning model to classify bird species based on their songs using audio processing and transformer-based models.

**Dataset:**  
- Audio files in FLAC format  
- 2 main components:
  - **Training data:** Audio files + labels (bird genus)
  - **Test data:** Audio files only (for prediction)

---

## 🛠 **Technical Implementation**

### 📂 **1. Data Preparation**
```python
# Unzip dataset
!unzip /content/birdsong-recognition-cmu.zip

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, Audio
```

**Label Mapping:**  
```python
GENUS_MAP = {
    'Acrocephalus':0, 'Anthus':1, 'Columba':2, 
    'Corvus':3, 'Emberiza':4, 'Motacilla':5,
    'Passer':6, 'Phylloscopus':7, 'Pluvialis':8,
    'Poecile':9, 'Streptopelia':10, 'Sylvia':11,
    'Tringa':12, 'Turdus':13
}
```

**Data Loading:**  
```python
# Load training data
train_df = pd.read_csv('/content/train.csv')
train_df['file_path'] = '/content/Data_files/train/xc' + train_df['file_id'].astype(str) + '.flac'
train_df['genus'] = train_df['genus'].map(GENUS_MAP)
```

---

### 🎚 **2. Audio Preprocessing**
**Create Audio Dataset:**  
```python
audio_dataset = Dataset.from_pandas(train_df)
audio_dataset = audio_dataset.rename_columns({
    "genus": "label", 
    "file_path": "file"
})
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))
```

**Feature Extraction:**  
```python
from transformers import AutoFeatureExtractor

MODEL_ID = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID, 
    do_normalize=True,
    return_attention_mask=True
)
```

---

### 🤖 **3. Model Architecture**
**Load Pre-trained Model:**  
```python
from transformers import AutoModelForAudioClassification

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID,
    num_labels=14  # 14 bird genera
)
```

**Training Configuration:**  
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="bird-song-classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=False
)
```

---

### 🏋️ **4. Model Training**
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=feature_extractor
)

trainer.train()
```

---

### 🔮 **5. Prediction Pipeline**
**Create Inference Pipeline:**  
```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification", 
    model="bird-song-classifier"
)
```

**Make Predictions:**  
```python
predictions = []
for audio_sample in test_dataset:
    result = classifier(audio_sample['audio']['array'])
    predictions.append(result[0]['label'])
```

---

## 📊 **Results & Output**
**Format Predictions:**  
```python
# Convert numeric labels back to genus names
INVERSE_MAP = {v:k for k,v in GENUS_MAP.items()}
test_df['genus'] = [INVERSE_MAP[int(pred.split('_')[1])] for pred in predictions]

# Save results
test_df[['file_id', 'genus']].to_csv('bird_song_predictions.csv', index=False)
```

---

## 🌟 **Key Features**
- 🎯 **High Accuracy:** Leveraging state-of-the-art audio transformers
- ⚡ **Efficient:** Using distilled model for faster inference
- 🔄 **Reproducible:** Complete pipeline from data to predictions

## 🚀 **Potential Improvements**
- 🔊 Audio augmentation for better generalization
- 📈 Hyperparameter tuning
- 🏷 Multi-label classification for species-level identification

---

<div style="text-align: center; margin-top: 20px;">
    <img src="https://media.giphy.com/media/3o7TKMt1VVNkHV2PaE/giphy.gif" width="200"/>
    <p style="font-style: italic;">Happy bird watching through audio! 🎧🐦</p>
</div>
