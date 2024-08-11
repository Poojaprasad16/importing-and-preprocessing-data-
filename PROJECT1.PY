# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the Iris dataset using pandas
dataset = pd.read_csv('iris.csv')

# Displaying the first few rows of the dataset to understand its structure
print("Dataset head:")
print(dataset.head())
print()

# Identifying features and dependent variable
# Features are all columns except the last one (target), which is the dependent variable
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values   

# Printing the matrix of features (X) and the dependent variable vector (y)
print("Matrix of features (X):")
print(X)
print()

print("Dependent variable vector (y):")
print(y)
