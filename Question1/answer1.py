# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Handle missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)  # Fill missing Age with median
train_data.drop(columns=['Cabin'], inplace=True)  # Drop Cabin column due to high missing values
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode

# Encode categorical variables
le_sex = LabelEncoder()
train_data['Sex'] = le_sex.fit_transform(train_data['Sex'])  # Encode Sex column

le_embarked = LabelEncoder()
train_data['Embarked'] = le_embarked.fit_transform(train_data['Embarked'])  # Encode Embarked column

# Define features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]  # Features
y = train_data['Survived']  # Target variable

# Scale continuous features
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])  # Scale Age and Fare

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Decision Tree (optimized grid)
param_grid_dt = {
    'max_depth': [3, 5, 10],  # Reduced range
    'min_samples_split': [2, 5],  # Fewer options
    'min_samples_leaf': [1, 2],  # Fewer options
    'criterion': ['gini']  # Single criterion
}

grid_search_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt,
    cv=3,  # Fewer folds
    scoring='f1',
    verbose=1,
    n_jobs=-1  # Use all available CPU cores
)
grid_search_dt.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params_dt = grid_search_dt.best_params_
print("Optimal Decision Tree Hyperparameters:", best_params_dt)

# Train the optimized Decision Tree model
best_tree = grid_search_dt.best_estimator_

# Make predictions
y_pred_dt = best_tree.predict(X_test)

# Evaluate the Decision Tree model
print("\nOptimized Decision Tree Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_dt):.2f}")

# Hyperparameter tuning for Random Forest (optimized grid)
param_grid_rf = {
    'n_estimators': [50, 100],  # Reduced range
    'max_depth': [3, 5, 10],  # Reduced range
    'min_samples_split': [2, 5],  # Fewer options
    'min_samples_leaf': [1, 2],  # Fewer options
    'criterion': ['gini']  # Single criterion
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    cv=3,  # Fewer folds
    scoring='f1',
    verbose=1,
    n_jobs=-1  # Use all available CPU cores
)
grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params_rf = grid_search_rf.best_params_
print("\nOptimal Random Forest Hyperparameters:", best_params_rf)

# Train the optimized Random Forest model
best_rf = grid_search_rf.best_estimator_

# Make predictions
y_pred_rf_optimized = best_rf.predict(X_test)

# Evaluate the Random Forest model
print("\nOptimized Random Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_optimized):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_optimized):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_optimized):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf_optimized):.2f}")

# Visualize confusion matrices
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf_optimized)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Decision Tree Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

# Visualize feature importances for both models
plt.figure(figsize=(12, 5))

# Feature importance for Decision Tree
feature_importances_dt = pd.Series(best_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.subplot(1, 2, 1)
sns.barplot(x=feature_importances_dt.values, y=feature_importances_dt.index)
plt.title('Feature Importances (Decision Tree)')
plt.xlabel('Importance')
plt.ylabel('Features')

# Feature importance for Random Forest
feature_importances_rf = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.subplot(1, 2, 2)
sns.barplot(x=feature_importances_rf.values, y=feature_importances_rf.index)
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')

plt.tight_layout()
plt.show()