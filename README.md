#  House Price Prediction using Regression Models

This project is based on a dataset from Kaggle and aims to predict the sale price of houses using machine learning models. The goal is to build a predictive model that can estimate house prices based on various features like lot size, number of rooms, quality rating, and more.

# Dataset
- Source: [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Features include: LotArea, OverallQual, YearBuilt, GrLivArea, GarageCars, etc.
- Target variable: `SalePrice`

# Steps Performed
1. **Exploratory Data Analysis (EDA)**  
   - Missing value heatmap  
   - Distribution of sale price  
   - Boxplots & scatterplots  
   - Correlation matrix

2. **Data Preprocessing**
   - Dropped columns with excessive missing values  
   - Filled missing numeric values with median  
   - Filled missing categorical values with mode  
   - One-hot encoded categorical features

3. **Model Building & Evaluation**  
   - **Linear Regression**  
   - **Random Forest Regressor**  
   - Evaluation metrics: RMSE and R² Score


## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python house_price_prediction.py  # or open the .ipynb file in Jupyter
