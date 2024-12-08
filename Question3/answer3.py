# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set style for seaborn visualizations
sns.set(style="whitegrid")

# Step 1: Load Dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Feature names
y = pd.Series(data.target)  # Target variable
print("Data loaded successfully.")
print(X.head())
print(y.head())

# Step 2: Data Preprocessing
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale the features
print("Features scaled successfully.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Step 3: Model Training and Hyperparameter Tuning
# Initialize models
linear_model = LinearRegression()
ridge = Ridge()
lasso = Lasso()
rf = RandomForestRegressor(random_state=42)

# Train models with their respective hyperparameter tuning

# Ridge Regression with GridSearchCV
ridge_params = {'alpha': np.logspace(-4, 4, 50)}
ridge_search = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2')
ridge_search.fit(X_train, y_train)
best_ridge = ridge_search.best_estimator_
print(f"Best Ridge alpha value: {ridge_search.best_params_['alpha']}")

# Lasso Regression with GridSearchCV
lasso_params = {'alpha': np.logspace(-4, 4, 50)}
lasso_search = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2')
lasso_search.fit(X_train, y_train)
best_lasso = lasso_search.best_estimator_
print(f"Best Lasso alpha value: {lasso_search.best_params_['alpha']}")

# Random Forest - set hyperparameters manually
rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'random_state': [42]}
rf_search = GridSearchCV(rf, rf_params, cv=5, scoring='r2')
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("Random Forest model trained with optimal hyperparameters.")

# Step 4: Evaluate Models
# Evaluate all models on test data
models = {
    "Linear Regression": linear_model,
    "Ridge Regression": best_ridge,
    "Lasso Regression": best_lasso,
    "Random Forest": best_rf
}

# Train Linear Regression on training data and evaluate
linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_preds))
linear_r2 = r2_score(y_test, linear_preds)

# Dictionary for comparison
results = {
    "Model": [],
    "RMSE": [],
    "R^2 Score": []
}

for name, model in models.items():
    # Train each model and evaluate predictions
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Log results
    results["Model"].append(name)
    results["RMSE"].append(rmse)
    results["R^2 Score"].append(r2)

# Convert to DataFrame for visualization
results_df = pd.DataFrame(results)
print("Model evaluation results:")
print(results_df)

# Plotting model comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results_df, palette="viridis")
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R^2 Score', data=results_df, palette="viridis")
plt.title("Model Comparison - R^2 Score")
plt.ylabel("R^2 Score")
plt.xticks(rotation=45)
plt.show()