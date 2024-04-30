import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from warnings import filterwarnings
#from xgboost import XGBRegressor


# Suppress warnings (optional)
filterwarnings('ignore')

# Assuming your data file is in the same directory as your script
data_path = "./ML_project/notebook/data/StudentsPerformance.csv"

# Read the CSV data
df = pd.read_csv(data_path)

# Data Cleaning and Preprocessing

# Print the first few rows
print(df.head())

# Check for missing values
print(df.isna().sum())

# Check for duplicate rows
print(df.duplicated().sum())

# Descriptive statistics (optional)
print(df.describe())

# Feature Engineering (optional)
#df.columns = df.columns.str.lower()  # Convert column names to lowercase for consistency
#df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']

# Separate Features and Target
X = df.iloc[:, :-1]  # All columns except the last (writing score)
y = df['writing score']

# Feature Preprocessing (One-Hot Encoding for categorical features, Standardization for numerical)
categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(exclude='object').columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories

preprocessor = ColumnTransformer(
    [
        ('OneHotEncoder', oh_transformer, categorical_features),
        ('StandardScaler', numeric_transformer, numerical_features)
    ]
)

X = preprocessor.fit_transform(X)

# Model Training and Evaluation

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(true, predicted):
    """
    Calculates evaluation metrics for regression models.
    """
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return mae, mse, rmse, r2


models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    #'XGBRegressor': XGBRegressor(),  # May require additional installation (pip install xgboost)
    'AdaBoost Regressor': AdaBoostRegressor()
}

model_list = []
r2_list = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, mse, rmse, r2 = evaluate(y_test, y_pred)

    print(f"\n**{model_name} Model Performance:**")
    print("- Mean Absolute Error: {:.4f}".format(mae))
    print("- Mean Squared Error: {:.4f}".format(mse))
    print("- Root Mean Squared Error: {:.4f}".format(rmse))
    print("- R2 Score: {:.4f}".format(r2))

    model_list.append(model_name)
    r2_list.append(r2)


# Check if lists are empty (optional)
if not model_list or not r2_list:
    print("Warning: model_list or r2_list is empty. DataFrame will be empty.")

# Create DataFrame using zip and dictionary comprehension
df = pd.DataFrame({
    'Model Name': model_list,
    'R2 Score': r2_list
})

# Sort by R2 Score
df = df.sort_values(by=['R2 Score'], ascending=False)

# Print the DataFrame
print(df)

line_model = LinearRegression(fit_intercept=True)
line_model = line_model.fit(X_train, y_train)
ypred = line_model.predict(X_test)
score = r2_score(y_test, ypred)*100
print('Accuracy of the model is %.2f' %score)

plt.scatter(y_test, ypred);
plt.xlabel('Actual');
plt.ylabel('Predicted');

sns.regplot(x=y_test, y=ypred, ci=None, color='green');

pred_df = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': ypred, 'Difference': y_test-ypred})
print(pred_df)
