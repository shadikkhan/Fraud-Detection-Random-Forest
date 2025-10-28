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
git clone https://github.com/shadikkhan/Fraud-Detection-Random-Forest.git
cd Fraud-Detection-Random-Forest
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

### Option 2: MyBinder (No Installation Required)

Click the **"launch binder"** badge at the top to run the notebook directly in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR_USERNAME/fraud-detection-project/HEAD?labpath=fraud_detection_random_forest.ipynb)

*Note: You'll need to upload the dataset manually in MyBinder*

### Option 3: Google Colab
1. Upload the notebook to Google Colab
2. Install requirements: `!pip install imbalanced-learn`
3. Upload the dataset using Colab's file upload feature

## 📖 Usage

### Quick Start
```python
# Basic usage example
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load and preprocess data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class balance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_resampled, y_resampled)
```

### Notebook Sections
1. **Data Loading & Exploration**: Dataset overview and basic statistics
2. **Data Preprocessing**: Feature scaling and preparation
3. **Class Imbalance Handling**: SMOTE implementation
4. **Model Training**: Random Forest with optimized parameters
5. **Model Evaluation**: Comprehensive metrics and visualization
6. **Results Analysis**: Confusion matrix and performance interpretation

## 📈 Model Performance

### Expected Results
- **Accuracy**: ~99.95%
- **Precision (Fraud)**: ~0.88
- **Recall (Fraud)**: ~0.85
- **F1-Score**: ~0.86
- **Training Time**: ~30-60 seconds (depending on hardware)

### Performance Optimizations
- **n_jobs=-1**: Utilizes all CPU cores for parallel processing
- **max_depth=10**: Prevents overfitting and reduces training time
- **min_samples_split=20**: Optimizes tree splitting for performance
- **SMOTE**: Addresses class imbalance effectively

## 📁 Project Structure

```
fraud-detection-project/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── fraud_detection_random_forest.ipynb    # Main analysis notebook
├── creditcard.csv                         # Dataset (not included in repo)
├── .gitignore                            # Git ignore rules
└── images/                               # Screenshots and plots (optional)
    └── confusion_matrix.png
```

## 🔧 Dependencies

### Core Libraries
- **pandas** (≥1.3.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scikit-learn** (≥1.0.0): Machine learning algorithms
- **imbalanced-learn** (≥0.8.0): SMOTE implementation

### Visualization
- **matplotlib** (≥3.5.0): Basic plotting
- **seaborn** (≥0.11.0): Statistical visualizations

### Development
- **jupyterlab** (≥3.0.0): Interactive development environment
- **notebook** (≥6.0.0): Jupyter notebook support

## 🔍 Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```bash
   FileNotFoundError: [Errno 2] No such file or directory: 'creditcard.csv'
   ```
   **Solution**: Download the dataset and place it in the project root directory

2. **Memory Issues**
   ```bash
   MemoryError: Unable to allocate array
   ```
   **Solution**: Use data sampling or increase system RAM

3. **Slow Training**
   ```bash
   RandomForestClassifier taking too long
   ```
   **Solution**: Reduce `n_estimators` or use the optimized parameters provided

4. **Import Errors**
   ```bash
   ModuleNotFoundError: No module named 'imblearn'
   ```
   **Solution**: Install missing packages: `pip install imbalanced-learn`


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- Credit Card Fraud Detection Dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- scikit-learn and imbalanced-learn communities
- Jupyter Project for the amazing notebook environment

