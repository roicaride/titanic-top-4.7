# 🚢 Titanic Survival Prediction ProjectnTop 4.7%

## Project Overview
This project implements multiple machine learning models to predict passenger survival on the Titanic. Using various features from the dataset, we've developed and compared different classification approaches to achieve the best possible prediction accuracy.

## Models Implemented
- Random Forest (84.13% accuracy)
- XGBoost (82.73% accuracy)
- Support Vector Machine (82.73% accuracy)
- Logistic Regression (82.45% accuracy)
- Neural Network (81.74% accuracy)

## Project Structure
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

## Features Engineering
- **Basic Features**: Age, Sex, Pclass, Fare
- **Family Features**: 
  - FamilySize (SibSp + Parch + 1)
  - IsAlone (Binary)
- **Name Features**:
  - Title extraction
  - HasParentheses
  - NameLength
- **Cabin Features**:
  - HasCabin (Binary)
- **Categorical Encodings**:
  - One-hot encoding for Pclass
  - One-hot encoding for Embarked
  - One-hot encoding for Title

## Data Preprocessing
1. Missing Value Handling:
   - Embarked: Filled with mode
   - Age: Iterative imputation using GradientBoostingRegressor
   - Cabin: Converted to binary feature
2. Feature Engineering
3. One-hot Encoding
4. Data Normalization

## Model Parameters

### Random Forest (Best Model)
- n_estimators: 200
- max_depth: 5
- min_samples_split: 2
- min_samples_leaf: 2
- max_features: log2
- bootstrap: False

### XGBoost
- n_estimators: 200
- max_depth: 7
- learning_rate: 0.05
- subsample: 0.6
- colsample_bytree: 0.4
- min_child_weight: 3
- gamma: 0.05

### Logistic Regression
- C: 10
- max_iter: 2000
- solver: lbfgs
- penalty: l2

### SVM
- C: 0.1
- kernel: linear
- gamma: auto

### Neural Network
- hidden_layer_sizes: (50,)
- activation: relu
- alpha: 0.1
- learning_rate_init: 0.005
- batch_size: auto

## Requirements
```python
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

## Usage
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the main script:
```bash
python titanic.py
```

## Results
- Best performing model: Random Forest (84.13% accuracy)
- All models achieved >81% accuracy
- Small performance spread (2.39% between best and worst models)
- Consistent predictions across different models

## Future Improvements
1. Feature Engineering:
   - Create more sophisticated family features
   - Extract more information from ticket numbers
   - Develop fare-based features

2. Model Optimization:
   - Implement stacking/ensemble methods
   - Further hyperparameter tuning
   - Test more complex neural network architectures

3. Data Processing:
   - Try different imputation strategies
   - Experiment with feature scaling methods
   - Implement feature selection techniques

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by Kaggle
- Inspired by various Kaggle kernels and discussions


