# 🚢 Titanic Top 4.7%

This repository presents a high-ranking solution to the legendary **Titanic - Machine Learning from Disaster** competition hosted on Kaggle.

By combining thoughtful feature engineering, robust preprocessing, and powerful model tuning techniques, this solution achieved a position in the **top 4.7% out of more than 16,000 competitors worldwide**.

---

## 💡 Project Highlights

- 📊 Full end-to-end pipeline: from raw data to Kaggle submission
- 🧠 Smart feature extraction: names, tickets, cabins, and group behavior
- 🛠️ Custom preprocessing: tailored pipelines for tree-based models
- 🔍 Thorough model selection: grid search on Random Forest and XGBoost
- 📈 Competitive result: top-tier placement in one of Kaggle’s most popular challenges

---

## 🧭 What You'll Find Here

- In-depth exploratory data analysis with visual insights
- Creative and domain-informed feature engineering
- Dual-pipeline architecture for flexible experimentation
- Hyperparameter tuning with `GridSearchCV`
- Reproducible code and clearly organized notebook

---

## 🔬 Feature Engineering Snapshot

Some of the crafted features that helped boost model performance:
- **Title extraction** from passenger names
- **NameLength** and presence of parentheses
- **Cabin-sharing patterns**
- **Shared tickets and travel groups**
- **Fare transformation** and family size logic
- One-hot encoded classes and boarding ports

---

## 🧪 Model Overview

Two pipelines were explored:
- **Pipeline A** – optimized for tree-based models (RandomForest, XGBoost)
- **Pipeline B** – designed for experimentation with scaling-sensitive models

After extensive tuning, **XGBoost** emerged as the top performer.

---

## 🏁 Final Result

- 🎯 **Top 4.7%** out of 16,000+ teams on Kaggle
- 📁 Submission ready and reproducible
- ✅ Clean, modular design with room for future improvements

---

## ⚙️ Tech Stack

- Python
- pandas & NumPy
- scikit-learn
- XGBoost
- seaborn & matplotlib

---

## 🚀 Future Enhancements

- SHAP-based model explainability
- Stacking/blending ensemble strategies
- Conversion to production-grade script or package
