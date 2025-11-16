# Brain Tumor MRI Classification using CNN  
### CRISP-DM Capstone Project — Deep Learning (TensorFlow/Keras)  
**Author:** Harshitha Mallappa  

---

## 📌 Project Overview  
This project builds an end-to-end **Brain Tumor MRI Classification System** using **Convolutional Neural Networks (CNNs)**.  
The model automatically classifies MRI brain scans as **Tumor** or **No Tumor**, supporting hospitals and radiologists through faster triage and improved diagnostic confidence.

This work follows the **CRISP-DM methodology** and includes:
- Business problem framing  
- Data understanding & exploratory data analysis (EDA)  
- Data cleaning & preprocessing  
- Baseline CNN model  
- Improved CNN model (Augmentation + Dropout)  
- Evaluation metrics  
- Confusion matrix  
- Executive summary & recommendations  

---

## 🎯 Business Problem
Hospitals face challenges such as:
- Long radiologist review times  
- High MRI scan volume  
- Risk of missed early tumor signs  

**Goal:**  
Use deep learning to reduce radiologist workload and assist in early tumor detection.

**Business Objective:**  
> Reduce average MRI review time by **20%** using AI-assisted screening.

---

## 📂 Dataset Details  
**Source:** Kaggle — Brain MRI Images Dataset  
**Classes:**
- **yes** → MRI contains a tumor  
- **no** → MRI shows no tumor  

**Total Images:**  
- Tumor (yes): **155**  
- No tumor (no): **98**  
- **Total: 253 images**

### 🔍 Dataset Characteristics
- Images vary widely in **size and brightness**
- Slight class imbalance (more tumor images)
- No metadata (age, gender, tumor type)

---

## 📊 Exploratory Data Analysis (EDA)

### Key Insights:
- Image sizes vary between **150–1900 pixels** → resizing required  
- Pixel brightness has heavy variation → normalization needed  
- Dataset moderately balanced → acceptable for binary classification  
- Visual inspection clearly shows structural differences between tumor vs no-tumor scans  

### Visuals Included:
- Class distribution bar chart  
- Sample MRI images  
- Image size distribution histogram  
- Brightness distribution histogram  
- Before–after preprocessing comparison  

---

## 🛠️ Data Preparation  
The following preprocessing steps were applied:

1. **Resizing** all images → 150×150  
2. **Normalization** using Rescaling(1/255)  
3. **Train–validation split** (80% / 20%)  
4. **Prefetching** for performance optimization  
5. **Data augmentation** (for improved model):
   - RandomFlip  
   - RandomRotation  
   - RandomZoom  

---

## 🤖 Modeling

### 1️⃣ Baseline CNN Model
Architecture includes:
- Rescaling  
- Conv2D + MaxPooling layers  
- Flatten  
- Dense layer  
- Sigmoid output  

**Purpose:** Establish a baseline performance.

---

### 2️⃣ Improved CNN Model  
Enhancements include:
- **Data Augmentation**  
- **Dropout layers** (0.3 & 0.5)  
- **Lower learning rate (1e-4)**  

**Purpose:** Reduce overfitting & improve validation performance.

---

## 📈 Evaluation

### ✔ Key Metrics (Improved Model)
- **Validation Accuracy:** 76%  
- **Precision (Tumor):** 78%  
- **Recall (Tumor):** 88%  
- **F1-Score:** 83%  

### ✔ Why Recall Matters  
In medical diagnosis, missing a tumor (false negative) can be life-threatening.  
A recall of **88%** means the model correctly detects *most* tumor cases.

### ✔ Confusion Matrix Included  
Shows strong ability to detect tumors vs non-tumor images.

---

## 💼 Business Impact

Using this model, hospitals can:
- Prioritize high-risk MRI scans  
- Reduce review delays  
- Improve early tumor detection  
- Reduce radiologist workload  
- Improve patient outcomes  

AI becomes a **clinical decision-support tool**, not a replacement.

---

## 🔮 What-If & Future Improvements

With more data & time, the system can be extended to:
- Classify different types of brain tumors  
- Perform tumor segmentation (pixel-level detection)  
- Use pretrained models (ResNet, VGG16, EfficientNet)  
- Apply Grad-CAM for explainability  
- Analyze 3D MRI scans  

---

## 📝 Files in This Repository
| File | Description |
|------|-------------|
| **Brain Tumor.ipynb** | Full notebook with EDA, preprocessing, models, evaluation |
| **Brain Tumor.html** | HTML version (for faculty review) |
| **README.md** | Project documentation |

---

## 🚀 How to Run Locally  
```bash
# Clone the repo
git clone https://github.com/h-m1998/brain-tumor-classification-CNN.git

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python

# Run notebook
jupyter notebook "Brain Tumor.ipynb"
