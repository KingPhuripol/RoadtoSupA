# 🧠 **EEG Data Classification Project**  
### *Classifying Brain Signals with Machine Learning*  

---

## 📌 **Project Overview**  
**Objective:** Build a machine learning model to classify EEG (Electroencephalogram) signals and predict whether a patient has neurological conditions (e.g., epilepsy).  

**Dataset:**  
- **Training Data:** 7,488 samples with 17 features (`V0`–`V15`, `id`) + target (`Class`)  
- **Test Data:** 7,488 samples (same features)  
- **Missing Values:**  
  - Training: 5,570  
  - Test: 12,923  

**Evaluation Metric:** **ROC AUC** (Area Under the Receiver Operating Characteristic Curve)  

---

## 🛠 **Methodology**  

### **1️⃣ Data Preparation**  
✅ Loaded & inspected data  
✅ Checked for missing values  
✅ Split into features (`X_train`) and target (`y_train`)  

### **2️⃣ Model Training (AutoML with AutoGluon)**  
🚀 Used **AutoGluon** to automate:  
- Model selection  
- Hyperparameter tuning  
- Ensemble learning  

```python
predictor = TabularPredictor(
    label='Class',
    eval_metric='roc_auc',
    problem_type='binary'
).fit(
    train_data,
    time_limit=600,  # 10 mins
    presets='best_quality'
)
```

### **3️⃣ Model Evaluation**  
🏆 **Best Model:** `WeightedEnsemble_L3` (**ROC AUC = 0.9744**)  

📊 **Leaderboard:**  

| Model                  | ROC AUC  | Fit Time (s) | Stack Level |
|------------------------|----------|--------------|-------------|
| WeightedEnsemble_L3    | 0.9744   | 400.18       | 3           |
| CatBoost_BAG_L2        | 0.9738   | 326.04       | 2           |
| LightGBMXT_BAG_L2      | 0.9737   | 324.54       | 2           |
| LightGBM_BAG_L2        | 0.9733   | 326.51       | 2           |

🔍 **Feature Importance:**  
*(Available if `predictor.feature_importance()` runs successfully)*  

### **4️⃣ Prediction & Submission**  
📤 **Prediction Distribution:**  
- **Class 0 (Normal):** 4,252 samples  
- **Class 1 (Abnormal):** 3,236 samples  

💾 **Saved Predictions:** `submission.csv`  

---

## 🎯 **Key Takeaways**  
✔ **High Accuracy:** Achieved **97.44% ROC AUC** with ensemble learning.  
✔ **Automated Workflow:** AutoGluon simplified model selection and tuning.  
✔ **Scalable:** Can handle larger datasets with more training time.  

---

## 🔄 **Next Steps**  
- **Handle Missing Values** (Imputation)  
- **Feature Engineering** (Extract more meaningful signals)  
- **Experiment with Deep Learning** (CNNs for EEG signals)  

---

### 🚀 **Let’s build smarter brainwave classifiers!**  
*(Code fully reproducible in Google Colab 🚀)*  
