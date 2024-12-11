# Import pandas library
import pandas as pd

# Load data
data = pd.read_csv('apartments_for_rent_classified.csv', sep=";", encoding='cp1252')

# Show the first five rows
data.head()

# columns, null values, data types
data.info()

# Show the columns
# Display column names
print("Columns:", data.columns)

# Display unique values for each column
for column in data.columns:
    print(f"Unique values in '{column}':\n", data[column].unique()[:10])  # Display first 10 unique values


# Set the target feature
target_column = 'category'
data[target_column].value_counts()


# Show the missing data
data.isnull().sum()

# Fill the missing values with the column mean 
for col in data.select_dtypes(include=['Float64', 'int64']).columns:
    data[col].fillna(data[col].median(), inplace=True)

# Fill the missing values with the column mode
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Check that the null values have been filled
data.isnull().sum()

# import libraries
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
label_encoder = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoder[col] = le

# Feature and target separation
X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data into training and test
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# %%
# Predictions
y_pred = model.predict(X_test)

# %%
# Model evaluation
"Accuracy:", accuracy_score(y_test, y_pred)
"Classification Report:\n", classification_report(y_test, y_pred)

# %%
# Interpret the results
import matplotlib.pyplot as plt

# %%
importance = model.feature_importances_
features = X.columns
sorted_indices = importance.argsort()

plt.figure(figsize=(20,6))
plt.bar(features[sorted_indices], importance[sorted_indices])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# %%
# Optimize the model
from sklearn.model_selection import GridSearchCV
para_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10,20,30],
    'min_samples_split': [2,5,10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), para_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
"Best parameters:", grid_search.best_params_

# %%
# Save the model for deployment
import joblib

joblib.dump(model, "apartment_classifier.pkl")
loaded_model = joblib.load("apartment_classifier.pkl")


