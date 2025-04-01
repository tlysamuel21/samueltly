# samueltly — Samuel Tsz Lam Yiu

Welcome to my personal portfolio and project site! I’m a Mathematics finalist at Imperial College London with a strong focus on **geophysical fluid dynamics**, **numerical simulation**, and **AI for environmental modeling**.

---

## 🌍 About Me

- 🎓 Final-year MSci Mathematics student (Dean’s List), Imperial College London  
- 💡 Passionate about climate modeling, fluid mechanics, and AI applications in Earth sciences  
- 🧠 Experience in supervised machine learning, numerical methods, and scientific computing  
- 🛠️ Languages: Python, R, MATLAB, C++, LaTeX  

---

## 🤖 AI & Scientific Machine Learning Projects

### 🔹 Methods for Data Science (Imperial College London)

This two-part coursework involved rigorous implementation of machine learning techniques without the use of high-level libraries like scikit-learn, focusing on both structured data and image-based scientific datasets.

#### **Coursework 1 – Supervised Learning from Scientific Data**

**Goal**: Predict electrical capacity of graphene-based electrodes and classify brain tumor types using fully custom models.

- 📊 **Regression Task**:  
  - Dataset: 558 graphene-based supercapacitor samples (12 features)  
  - Built models from scratch: **Decision Trees**, **Random Forests**, and **Gradient Boosted Trees**  
  - Implemented custom 5-fold cross-validation and evaluated models with **MSE** and **R²**  
  - Analysed feature importance using **Out-of-Bag (OOB)** estimation

- 🧠 **Multi-Layer Perceptron**:  
  - Developed a fully vectorized **2-hidden-layer MLP** in NumPy  
  - Compared optimizers: SGD, SGD with momentum, and **Nesterov’s Accelerated Gradient**  
  - Conducted convergence testing and hyperparameter tuning

- 🧪 **Classification Task**:  
  - Dataset: Brain tumor diagnosis (3-class classification)  
  - Implemented **k-NN**, **weighted k-NN**, and **2-step hierarchical k-NN**  
  - Designed and trained **logistic regression** and **kernel logistic regression** from scratch  
  - Used AUC-PR and precision/recall metrics to evaluate classifier sensitivity to minority classes

📎 [See Coursework 1 notebook](https://github.com/tlysamuel21/samueltly/blob/main/02079104_Coursework1.ipynb)

---

#### **Coursework 2 – Deep Learning & Manifold Learning**

**Goal**: Classify star types from telescope images and analyze geometric structure of latent feature spaces.

- 🌌 **Convolutional Neural Network for Star Classification**  
  - Dataset: 648 stellar images (Euclid telescope, 32×32 grayscale)  
  - Built a **custom CNN in PyTorch** with convolution, max-pooling, ReLU, and softmax layers  
  - Integrated **early stopping**, **L2 regularization**, and compared against class imbalance techniques:  
    - Reweighted loss  
    - Data augmentation (random rotations)

- 📉 **Dimensionality Reduction of High-Dimensional Embeddings**  
  - Reduced 180D feature vectors using:  
    - **Principal Component Analysis (PCA)**  
    - **Graph-based Isomap** with cosine distances and spectral embeddings  
  - Visualized class separability and evaluated cluster quality with the **Davies-Bouldin Index**  
  - Analyzed distance metrics: cosine vs resistance distances

📎 [See Coursework 2 notebook](https://github.com/tlysamuel21/samueltly/blob/main/02079104_Coursework2_Part1.ipynb)

---

### 🌞 ClimateHack.AI 2023 – DOXA AI Hackathon

Finalist in an international ML competition focused on solar energy forecasting.

- Developed a **CNN model** to predict cloud coverage and sunlight for solar power optimization  
- Top performer and **selected to represent Imperial College London** in the finals  
- Worked with real climate satellite imagery and used AI to improve clean energy outcomes

🔗 [ClimateHack.AI](https://climatehack.ai/#:~:text=Find%20out%20about%20our%20progress,carbon%20emissions%20with%20machine%20learning.)

---

### 🌡️ ECMWF: AI in Weather & Climate

- Enrolled in ECMWF’s **"Machine Learning for Weather & Climate"** online course  
- Topics include: data assimilation, nowcasting, and hybrid physics-AI modeling  
- Completed Tier 1 and progressing through Tiers 2 & 3

🛰️ **TrajGRU Nowcasting Project**  
As part of Tier 3 (nowcasting), I implemented a radar and lightning-based precipitation forecasting notebook using pretrained **TrajGRU models** provided by FBK and Arpae.

📎 [NWC_model.ipynb – TrajGRU Nowcasting Example](https://tlysamuel21.github.io/NWC-model.ipynb)

- Compared 1-channel (radar only) vs. 2-channel (radar + lightning) models  
- Added custom **forecast animations** and **MSE-based evaluation**  
- Explored storm evolution and reflected on forecast quality  
- This notebook serves as a **representative example** of my implementation and understanding of ECMWF course content.


> *All model/data credit to ECMWF, FBK, and Arpae.*

---

## 🧪 Award-Winning Research Project

### 🥇 _Best Second-Year Group Project_  
**An Investigation of Shearing Flows of Newtonian and Non-Newtonian Fluids**

- Modeled Couette and Poiseuille flows using Navier-Stokes and Oldroyd-B models  
- Implemented simulations for shear-thinning and shear-thickening fluids  
- Developed a simplified model for **blood flow under oscillating pressure**  
- Presented the **Weissenberg effect** and contrasted viscoelastic vs Newtonian behaviors

📄 [Read the PDF](https://github.com/tlysamuel21/samueltly/blob/main/M2R_Final_Report%20(1).pdf)

---

Thanks for visiting — I’m currently looking for **AI + climate-focused opportunities**, especially those aligned with environmental modeling, forecasting, or scientific data analysis.
