# Credit Card Fraud Detection with Random Forest

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## 🎯 Overview

This project implements a machine learning solution for credit card fraud detection using Random Forest classifier. The model addresses class imbalance using SMOTE (Synthetic Minority Oversampling Technique) and provides comprehensive evaluation metrics including confusion matrix visualization.

### Key Highlights
- **High Accuracy**: Achieves 99%+ accuracy on fraud detection
- **Balanced Approach**: Uses SMOTE to handle severely imbalanced dataset
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Production Ready**: Optimized Random Forest parameters for performance

## ✨ Features

- 🔍 **Data Preprocessing**: StandardScaler normalization for optimal model performance
- ⚖️ **Class Imbalance Handling**: SMOTE implementation for balanced training
- 🌲 **Random Forest Classification**: Optimized hyperparameters for speed and accuracy
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, and Support
- 🎨 **Visual Analysis**: Colorful confusion matrix heatmap
- 📈 **Performance Monitoring**: Detailed classification reports

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset** loaded directly from TensorFlow's data repository:
- **Data Source**: `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`
- **284,807 transactions** over 2 days in September 2013
- **492 fraudulent transactions** (0.172% of all transactions)
- **30 features**: 28 anonymized (V1-V28), Time, Amount, and Class
- **Highly imbalanced**: ~99.83% legitimate vs ~0.17% fraudulent

### Data Features
- `Time`: Seconds elapsed between transactions
- `V1-V28`: Anonymized features from PCA transformation
- `Amount`: Transaction amount
- `Class`: Target variable (0 = legitimate, 1 = fraud)

## 🚀 Setup & Installation

### Option 1: Local Setup (Recommended for Development)

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

#### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-project.git
cd fraud-detection-project
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv fraud_detection_env

# Activate virtual environment
# On macOS/Linux:
source fraud_detection_env/bin/activate
# On Windows:
fraud_detection_env\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Dataset Loading
The dataset is automatically loaded from TensorFlow's data repository via URL:
```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
```
No manual download required - the notebook handles data loading automatically.

#### Step 5: Launch Jupyter Notebook
```bash
# Start Jupyter Lab (recommended)
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

#### Step 6: Run the Analysis
1. Open `fraud_detection_random_forest.ipynb`
2. Run all cells sequentially
3. Review results and visualizations


## 📖 Usage

### Quick Start
```python
# Basic usage example - matches the actual implementation
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Load data directly from URL
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Prepare features and target
X = df.drop('Class', axis=1).values
y = df['Class'].values

# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE for class balance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest with optimized parameters
model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=20,           
    min_samples_split=10,   
    min_samples_leaf=5,    
    max_features='sqrt',    
    n_jobs=-1,             
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)
```

### Notebook Sections
1. **Package Installation**: Installing required dependencies (numpy, pandas, scikit-learn, scipy, pytest, imbalanced-learn, matplotlib, seaborn)
2. **Data Loading & Exploration**: Loading dataset from TensorFlow URL and basic statistics
3. **Data Preprocessing**: Feature scaling with StandardScaler and stratified train-test split
4. **Class Imbalance Analysis**: Detailed fraud vs non-fraud transaction counts and percentages
5. **SMOTE Implementation**: Synthetic minority oversampling for balanced training data
6. **Model Training**: Random Forest with optimized hyperparameters
7. **Model Evaluation**: Comprehensive metrics including classification report and confusion matrix
8. **Visualization**: Confusion matrix heatmap with matplotlib and seaborn

## 📈 Model Performance

### Expected Results
- **Accuracy**: ~99.95%
- **Precision (Fraud)**: ~0.88
- **Recall (Fraud)**: ~0.85
- **F1-Score**: ~0.86
- **Training Time**: ~30-60 seconds (depending on hardware)

### Performance Optimizations
- **n_jobs=-1**: Utilizes all CPU cores for parallel processing
- **max_depth=20**: Optimized depth to prevent overfitting while maintaining performance
- **min_samples_split=10**: Optimizes tree splitting for performance
- **min_samples_leaf=5**: Ensures sufficient samples in leaf nodes
- **max_features='sqrt'**: Uses square root of features for optimal performance
- **SMOTE**: Addresses class imbalance effectively with 50-50 distribution in training data

## 📁 Project Structure

```
fraud-detection-project/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── fraud_detection_random_forest.ipynb    # Main analysis notebook
├── fraud_detection_random_forest-v1.ipynb # Previous version
├── fraud_detection_nn.ipynb              # Neural network implementation
├── creditcard123.csv                     # Local dataset copy (optional)
├── data/                                 # Data directory
└── .gitignore                           # Git ignore rules
```

## 🔧 Dependencies

### Core Libraries
- **numpy** (≥1.21.0): Numerical computing and array operations
- **pandas** (≥1.3.0): Data manipulation and analysis
- **scikit-learn** (≥1.0.0): Machine learning algorithms and preprocessing
- **imbalanced-learn** (≥0.8.0): SMOTE implementation for handling class imbalance
- **scipy** (≥1.7.0): Scientific computing utilities
- **pytest** (≥6.0.0): Testing framework

### Visualization
- **matplotlib** (≥3.5.0): Basic plotting and visualization
- **seaborn** (≥0.11.0): Statistical visualizations and heatmaps

### Development
- **jupyterlab** (≥3.0.0): Interactive development environment
- **notebook** (≥6.0.0): Jupyter notebook support

## 🔍 Troubleshooting

### Common Issues

1. **Dataset Loading Issues**
   ```bash
   URLError: <urlopen error [Errno 8] nodename nor servname provided>
   ```
   **Solution**: Check internet connection or use a local copy of the dataset

2. **Memory Issues**
   ```bash
   MemoryError: Unable to allocate array
   ```
   **Solution**: Increase system RAM or use data sampling techniques

3. **Package Installation Errors**
   ```bash
   ModuleNotFoundError: No module named 'imblearn'
   ```
   **Solution**: Install missing packages: `pip install imbalanced-learn`

4. **Slow Training Performance**
   ```bash
   RandomForestClassifier taking too long
   ```
   **Solution**: The optimized parameters should provide good performance, but you can reduce `n_estimators` for faster training


**Happy Fraud Detection! 🔍💳**