import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings (optional)
warnings.filterwarnings('ignore')


data_path = "./ML_project/notebook/data/StudentsPerformance.csv"

# Read the CSV data
df = pd.read_csv(data_path)

# Print the first few rows
print(df.head())

# Get data shape (number of rows, columns)
print(df.shape)

# Get data information (data types, non-null values)
print(df.info())

# Check for missing values
print(df.isna().sum())

# Check for duplicate rows
print(df.duplicated().sum())

# Get number of unique values in each column
print(df.nunique())

# Descriptive statistics (mean, standard deviation, etc.)
print(df.describe())

# Separate numerical and categorical features
numerical_features = [col for col in df.columns if df[col].dtype != 'object']
categorical_features = [col for col in df.columns if df[col].dtype == 'object']

# Print feature lists
print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Feature Engineering (create new features)
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['average'] = df['total_score'] / 3

# Check average score distribution for students with 100 in math
print(df[df['math score'] == 100]['average'].count())

# Check average score distribution for students with math score <= 40
print(df[df['math score'] <= 40]['average'].count())

# Distribution of average score
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.distplot(df['average'], bins=30, kde=True, color='g')
plt.title("Distribution of Average Score")
plt.subplot(122)
sns.distplot(df['average'], kde=True, label='gender')
plt.title("Average Score by Gender")
plt.show()

# Distribution of total score
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
plt.subplot(121)
sns.distplot(df['total_score'], bins=30, kde=True, color='g')
plt.title("Distribution of Total Score")
plt.subplot(122)
sns.distplot(df['total_score'], kde=True, label='gender')
plt.title("Total Score by Gender")
plt.show()

# Correlation matrix for numerical features
data_corr = df.iloc[:, 4:8]  # Assuming numerical features are at indices 4 to 7

# Encode categorical variable "test preparation course" (optional)
data_corr['test preparation course'] = data_corr['test preparation course'].replace({'none': 0, 'completed': 1})

# Calculate correlation coefficients
data_corr = data_corr.corr()

# Heatmap for correlations
plt.figure(figsize=(8, 6))
sns.heatmap(data=data_corr, annot=True, cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

# Average score distribution by lunch type
plt.subplots(1, 3, figsize=(25, 6))
plt.subplot(141)
sns.distplot(df['average'], kde=True, label='lunch')
plt.title("Average Score by Lunch")
plt.subplot(142)
sns.distplot(df[df['gender'] == 'female']['average'], kde=True, label='lunch')
plt.title("Average Score (Female) by Lunch")
plt.subplot(143)
sns.distplot(df[df['gender'] == 'male']['average'], kde=True, label='lunch')
plt.title("Average Score (Male) by Lunch")
plt.show()