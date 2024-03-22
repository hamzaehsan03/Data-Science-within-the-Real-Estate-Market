import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('housing.csv')
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Delete rows with missing values
df.dropna(inplace=True)

sns.set(style='whitegrid', context='notebook')

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

df = pd.get_dummies(df, columns=['Neighborhood'])

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

