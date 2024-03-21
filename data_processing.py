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

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Price', kde=True)
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
