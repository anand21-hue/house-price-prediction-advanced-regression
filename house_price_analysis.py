import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Load the data
df = pd.read_csv('train.csv')

# Check info and basic statistics
print("Train Data Info:")
print(df.info())

print("\nTrain Data Description:")
print(df.describe())

# Missing values heatmap
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.show()

# SalePrice distribution
sns.histplot(df['SalePrice'], kde=True)
plt.show()

# Boxplot: OverallQual vs SalePrice
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.show()

# Scatterplot: GrLivArea vs SalePrice
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.show()

# Correlation heatmap for SalePrice
corr = df.corr(numeric_only=True)
sns.heatmap(corr[['SalePrice']].sort_values(by='SalePrice', ascending=False), annot=True, cmap='coolwarm')
plt.show()

# Drop columns with too many missing values
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
# Fill numeric NaNs with median
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical NaNs with mode
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
 
# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Train-test split (corrected syntax)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# ... (previous code remains the same until evaluation) ...

# Linear Regression Evaluation
y_pred = model.predict(X_test)
# Calculate RMSE manually
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = (mse_lr)**0.5  # Manual square root calculation
print('Linear Regression RMSE:', rmse_lr)
print('Linear Regression R²:', r2_score(y_test, y_pred))
# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Calculate RMSE manually
mse_rf = mean_squared_error(y_test, rf_pred)
rmse_rf = (mse_rf)**0.5  # Manual square root calculation
print('\nRandom Forest RMSE:', rmse_rf)
print('Random Forest R²:', r2_score(y_test, rf_pred))
