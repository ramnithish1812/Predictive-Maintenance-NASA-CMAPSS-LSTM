# ✈️ Predictive Maintenance using NASA C-MAPSS FD001 and LSTM Networks

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## 📌 Project Overview

Predictive Maintenance is a critical component of modern Industry 4.0 systems, enabling organizations to anticipate equipment failures before they occur. This project develops a deep learning-based Remaining Useful Life (RUL) prediction system using NASA's C-MAPSS FD001 turbofan engine dataset.

An LSTM (Long Short-Term Memory) network is trained on multivariate sensor measurements to learn engine degradation patterns and estimate the number of cycles remaining before failure.

The system can assist maintenance teams in reducing downtime, improving safety, and optimizing maintenance schedules through data-driven decision making.

---

## 🎯 Objectives

* Predict Remaining Useful Life (RUL) of aircraft turbofan engines
* Learn degradation patterns from multivariate sensor data
* Implement an end-to-end predictive maintenance pipeline
* Apply deep learning techniques to industrial prognostics
* Demonstrate real-world applications of sequence modeling using LSTMs

---

## 📊 Dataset

### NASA C-MAPSS FD001 Dataset

The Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset was developed by NASA for Prognostics and Health Management (PHM) research.

#### Dataset Characteristics

| Feature              | Value                                      |
| -------------------- | ------------------------------------------ |
| Training Engines     | 100                                        |
| Test Engines         | 100                                        |
| Sensor Measurements  | 21                                         |
| Operational Settings | 3                                          |
| Total Features       | 26                                         |
| Fault Mode           | High Pressure Compressor (HPC) Degradation |
| Operating Conditions | 1                                          |

Each engine starts in a healthy state and gradually degrades until failure.

---

## 🏗️ System Architecture

![Architecture](images/architecture.png)

### Workflow

```text
NASA C-MAPSS Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Preprocessing
        │
        ▼
RUL Generation
        │
        ▼
Feature Scaling
        │
        ▼
Sequence Creation
        │
        ▼
Train / Validation Split
        │
        ▼
LSTM Network
        │
        ▼
Model Training
        │
        ▼
Evaluation
        │
        ▼
RUL Prediction
```

---

## 🔧 Data Preprocessing

### 1. Data Loading

* Imported NASA FD001 dataset
* Assigned meaningful column names
* Converted raw text files into Pandas DataFrames

### 2. Remaining Useful Life (RUL) Generation

Target labels were generated using:

```python
RUL = Max_Cycle - Current_Cycle
```

### 3. Feature Scaling

Applied MinMaxScaler to normalize sensor values between 0 and 1.

### 4. Sequence Generation

Created sliding windows of 30 cycles for LSTM input.

Input Shape:

```python
(samples, timesteps, features)
```

Example:

```python
(10000, 30, 24)
```

---

## 🤖 Model Architecture

```python
model = Sequential([
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])
```

### Architecture Summary

| Layer  | Configuration   |
| ------ | --------------- |
| LSTM   | 64 Units        |
| Dense  | 32 Units + ReLU |
| Output | 1 Neuron        |

### Why LSTM?

LSTM networks are particularly effective for:

* Time-series forecasting
* Sequential pattern recognition
* Learning long-term dependencies
* Remaining Useful Life estimation

---

## ⚙️ Training Configuration

| Parameter     | Value               |
| ------------- | ------------------- |
| Optimizer     | Adam                |
| Loss Function | Mean Squared Error  |
| Metric        | Mean Absolute Error |
| Epochs        | 20                  |
| Batch Size    | 64                  |

---

## 📈 Results

### Sample Predictions

```text
135.62
193.31
177.72
140.79
67.97
```

### Prediction Performance

![Prediction Results](images/predictions.png)

The scatter plot compares actual RUL values with model predictions. Points closer to the diagonal line indicate higher prediction accuracy.

---

## 📁 Repository Structure

```text
Predictive-Maintenance-NASA-CMAPSS-LSTM/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── notebooks/
│   └── NASA_FD001_LSTM.ipynb
│
├── models/
│   └── fd001_lstm_model.keras
│
├── docs/
│   └── Damage_Propagation_Modeling.pdf
│
├── images/
│   ├── architecture.png
│   ├── predictions.png
│   ├── sample_predictions.png
│   └── model_summary.png
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Predictive-Maintenance-NASA-CMAPSS-LSTM.git
```

Navigate into the project:

```bash
cd Predictive-Maintenance-NASA-CMAPSS-LSTM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Google Colab

---

## 📚 Reference

**Saxena, A., Goebel, K., Simon, D., & Eklund, N.**

*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*

NASA Prognostics and Health Management (PHM) Conference, 2008.

---

## 🔮 Future Enhancements

* Bidirectional LSTM Networks
* GRU-based Architectures
* Transformer Models
* Explainable AI (XAI)
* Hyperparameter Optimization
* Real-Time Deployment Dashboard
* Streamlit Web Application

---

## 👨‍💻 Author

**Ram**

Machine Learning • Deep Learning • Predictive Maintenance • Prognostics & Health Management

---

