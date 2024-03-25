"""
This script performs data processing and analysis on a housing dataset.
It includes loading the data, handling missing values and duplicates, visualizing data through graphs,
performing feature engineering, and training and evaluating machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the data
df = pd.read_csv('housing.csv')
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Delete rows with missing values
df.dropna(inplace=True)

# Check for duplicates
print(df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

sns.set_theme(style='whitegrid', context='notebook')

graphs = input('Do you want to see the graphs? (y/n): ')

if graphs == 'y':
    # Distribution of Property Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Price', kde=True)
    plt.title('Distribution of Property Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    # Price vs Square Feet
    plt.figure
    sns.scatterplot(data=df, x='SquareFeet', y='Price')
    plt.title('Price vs Square Feet')
    plt.xlabel('Square Feet')
    plt.ylabel('Price')
    plt.show()

    # Price vs Number of Bedrooms
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Bedrooms', y='Price')
    plt.title('Price Distribution by Number of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price')
    plt.show()

    # Price vs Number of Bathrooms
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Bathrooms', y='Price')
    plt.title('Price Distribution by Number of Bathrooms')
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Price')
    plt.show()

    # Price by Location
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Neighborhood', y='Price')
    plt.title('Price Distribution by Location')
    plt.xlabel('Neighborhood')
    plt.ylabel('Price')
    plt.show()

else:
    pass

df = pd.get_dummies(df, columns=['Neighborhood'])

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

X = df.drop(['Price'], axis=1) 
y = df['Price'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=727)

param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees in random forest
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'max_depth': [10, 20, 30],         # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],     # Minimum number of samples required at each leaf node
    'learning_rate': [0.01, 0.1, 0.2]  # Learning rate
}


gb_regressor = GradientBoostingRegressor(random_state=727)

grid_search = GridSearchCV(estimator=gb_regressor, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters found
print(grid_search.best_params_)

# Best estimator
best_grid = grid_search.best_estimator_

# Make predictions using the best returned model
y_pred_best = best_grid.predict(X_test)


rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
print(f"Best Grid RMSE: {rmse_best}")

r2_best = r2_score(y_test, y_pred_best)
print(f"Best Grid R-squared: {r2_best}")

mae = mean_absolute_error(y_test, y_pred_best)
print(f"Mean Absolute Error: {mae}")

mse = mean_squared_error(y_test, y_pred_best)
print(f"Mean Squared Error: {mse}")

variance = np.var(y_pred_best)
print(f"Variance of Predictions: {variance}")

'''
{'learning_rate': 0.01, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}
Best Grid RMSE: 50785.9969751395
Best Grid R-squared: 0.5583561322583783
'''

# gb_regressor = GradientBoostingRegressor(random_state=727)
# gb_regressor.fit(X_train, y_train)
# y_pred_gb = gb_regressor.predict(X_test)

# rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
# r2_gb = r2_score(y_test, y_pred_gb)
# print(f"Gradient Boosting RMSE: {rmse_gb}")
# print(f"Gradient Boosting R-squared: {r2_gb}")

''' Random Forest Model 
X = df.drop(['Price'], axis=1) # Features
y = df['Price'] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=727) # wysi

rf_regressor = RandomForestRegressor(random_state=727)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest R-squared: {r2_rf}")
'''


''' Linear Regression Model with Polynomial Features
X = df.drop(['Price'], axis=1) # Features
y = df['Price'] # Target
# Create polynomial features
poly = PolynomialFeatures(degree=2) # Experiment with different degrees
X_poly = poly.fit_transform(X)

# Split the new polynomial features into training and testing sets
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=727)

# Fit the Linear Regression model on the polynomial features
regressor_poly = LinearRegression()
regressor_poly.fit(X_train_poly, y_train)
y_pred_poly = regressor_poly.predict(X_test_poly)

# Evaluate the new model
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly = r2_score(y_test, y_pred_poly)
print(f"Polynomial RMSE: {rmse_poly}")
print(f"Polynomial R-squared: {r2_poly}")
'''


''' Linear Regression Model 
X = df.drop(['Price'], axis=1) # Features
y = df['Price'] # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=727) # wysi

# Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test) 

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

explained_variance = explained_variance_score(y_test, y_pred)
print(f"Explained Variance Score: {explained_variance}")

print(f"Mean Squared Error: {mse}")
'''
