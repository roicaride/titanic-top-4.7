# Titanic Survival Prediction TOP 4.7% Kaggle competition🚢

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.2-red.svg)](https://xgboost.readthedocs.io/)

## Overview 📊

This project implements multiple machine learning models to predict passenger survival on the Titanic. Using various features from the dataset, we've developed and compared different classification approaches to achieve the best possible prediction accuracy.

## Model Performance 📈

| Model | Accuracy |
|-------|----------|
| Random Forest | 84.13% |
| XGBoost | 82.73% |
| SVM | 82.73% |
| Logistic Regression | 82.45% |
| Neural Network | 81.74% |

## Project Structure 📁

```
titanic/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── best_random_forest.joblib
│   ├── best_xgboost.joblib
│   ├── best_svm.joblib
│   ├── best_logistic_regression.joblib
│   └── best_neural_network.joblib
│
├── titanic.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

## Features Engineering 🛠️

### Basic Features
- Age
- Sex
- Pclass
- Fare

### Engineered Features
- **Family Features**
  - FamilySize (SibSp + Parch + 1)
  - IsAlone (Binary)
- **Name Features**
  - Title extraction
  - HasParentheses
  - NameLength
- **Cabin Features**
  - HasCabin (Binary)
- **Categorical Encodings**
  - One-hot encoding for Pclass
  - One-hot encoding for Embarked
  - One-hot encoding for Title

## Model Parameters 🎯

### Random Forest (Best Model)
```python
{
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'bootstrap': False
}
```

### XGBoost
```python
{
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.6,
    'colsample_bytree': 0.4,
    'min_child_weight': 3,
    'gamma': 0.05
}
```

## Installation 💻

1. Clone the repository
```bash
git clone https://github.com/yourusername/titanic.git
cd titanic
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Open the Jupyter notebook:
```bash
jupyter notebook titanic.ipynb
```

2. Run all cells to:
   - Load and preprocess data
   - Train models
   - Generate predictions
   - Visualize results

## Results 📊

- Best performing model: Random Forest (84.13% accuracy)
- All models achieved >81% accuracy
- Small performance spread (2.39% between best and worst models)
- Consistent predictions across different models

## Future Improvements 🔮

1. Feature Engineering:
   - Create more sophisticated family features
   - Extract more information from ticket numbers
   - Develop fare-based features

2. Model Optimization:
   - Implement stacking/ensemble methods
   - Further hyperparameter tuning
   - Test more complex neural network architectures

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Dataset provided by [Kaggle](https://www.kaggle.com/c/titanic)
- Inspired by various Kaggle kernels and discussions 
