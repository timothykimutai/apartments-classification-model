# Apartment Classification Project

## Overview
This project aims to classify apartments based on provided features using machine learning techniques. The dataset includes various attributes of apartments and a target feature representing the category or class for classification. The Random Forest algorithm is used for model training and evaluation, with hyperparameter tuning for optimization.

---

## Project Structure
The project is structured as follows:

```
Apartment_Classification_Project/
|
|-- data/
|   |-- apartments_for_rent_classified.csv   # Dataset
|
|-- notebooks/
|   |-- EDA_and_Preprocessing.ipynb              # Exploratory data analysis and preprocessing
|   |-- Model_Training_and_Evaluation.ipynb      # Model building and evaluation
|
|-- src/
|   |-- preprocessing.py                         # Preprocessing functions
|   |-- model_training.py                        # Model training and optimization
|
|-- models/
|   |-- apartment_classifier.pkl                 # Saved trained model
|
|-- README.md                                    # Project documentation
```

---

## Steps

### 1. Data Loading and Exploration
- **Objective**: Understand the dataset structure, check for missing values, and identify the target column.
- **Tools**: `pandas`
- **Key Outputs**: Dataset summary, unique values in each column, missing value analysis.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv("apartments_for_rent_classified.csv")
data.head()
data.info()
data.describe()
```

### 2. Data Preprocessing
- **Objective**: Clean and preprocess the dataset for modeling.
- **Steps**:
  - Handle missing values.
  - Encode categorical features.
  - Split the data into training and testing sets.
- **Tools**: `pandas`, `sklearn`

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Example preprocessing steps
le = LabelEncoder()
data['categorical_column'] = le.fit_transform(data['categorical_column'])
X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Model Training and Evaluation
- **Objective**: Train a classification model and evaluate its performance.
- **Model**: Random Forest Classifier
- **Metrics**: Accuracy, Classification Report

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### 4. Hyperparameter Tuning
- **Objective**: Optimize model parameters for better performance.
- **Tool**: GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
```

### 5. Feature Importance
- **Objective**: Identify the most significant predictors.
- **Tool**: Matplotlib

```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns
sorted_indices = importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(features[sorted_indices], importances[sorted_indices])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### 6. Model Deployment
- **Objective**: Save the trained model for deployment.
- **Tool**: `joblib`

```python
import joblib

joblib.dump(model, "models/apartment_classifier.pkl")
```

---

## Requirements

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

**Required Libraries**:
- pandas
- scikit-learn
- matplotlib
- joblib

---

## Future Improvements
- Implement additional models (e.g., XGBoost, SVM) for comparison.
- Explore feature engineering to enhance predictive power.
- Integrate the project into a web application using Flask or Streamlit.

---

## Author
Developed by TIMOTHY KIMUTAI. Feel free to reach out for any queries or suggestions.

