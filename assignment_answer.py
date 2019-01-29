# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading the dataset
dataset = pd.read_excel('./data/HousingPrice.xls')

# Define unused columns
ununsed_columns = ["Order", "PID"]

dataset = dataset.drop(ununsed_columns, axis = 1)
# dataset = dataset.dropna(how = 'any', axis = 0)

price = dataset["SalePrice"]

dataset = dataset.drop(['SalePrice'], axis = 1)

dataset.info()
dataset.describe()

# Separating the dataset and output
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 21].values

# Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
dataset["MS Zoning"] = labelencoder_X.fit_transform(dataset["MS Zoning"])
dataset["Lot Shape"] = labelencoder_X.fit_transform(dataset["Lot Shape"])
dataset["Utilities"] = labelencoder_X.fit_transform(dataset["Utilities"])
dataset["Condition 1"] = labelencoder_X.fit_transform(dataset["Condition 1"])
dataset["Condition 2"] = labelencoder_X.fit_transform(dataset["Condition 2"])
dataset["Bldg Type"] = labelencoder_X.fit_transform(dataset["Bldg Type"])
dataset["House Style"] = labelencoder_X.fit_transform(dataset["House Style"])
dataset["Foundation"] = labelencoder_X.fit_transform(dataset["Foundation"])
dataset["Bsmt Qual"] = labelencoder_X.fit_transform(dataset["Bsmt Qual"])
dataset["Central Air"] = labelencoder_X.fit_transform(dataset["Central Air"])
dataset["Kitchen Qual"] = labelencoder_X.fit_transform(dataset["Kitchen Qual"])

# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, price, test_size=0.20, random_state=42)

# Fitting MLR to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
