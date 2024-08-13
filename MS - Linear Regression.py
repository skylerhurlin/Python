import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# For this assignment, I used a dataset containing body measurements of customers for an online clothing retailer. The objective for this linear regression is to use Weight measurements to predict Chest measurements for male customers.

d1 = pd.read_csv('C:/Users/skyef/Documents/homework/data/BodyMeas.csv')
print(d1.head())
d1.info()

# What is the gender split? Remove female customers from the dataset.

print(d1.Gender.value_counts())

d1M = d1[d1['Gender'].str.contains('M')]

# Use a scatterplot to examine the distribution of Weight and Chest measurements. Using transparency makes it easier to see overlapping points. 

sns.scatterplot(d1M, x='Weight', y='Chest', alpha=0.5)
plt.show()

# Using the describe function gives the mean, stdev, and distribution numbers.

d1M[['Weight','Chest']].describe()

# I will remove the most dramatic outliers by setting the upper bound for Weight at 325.

d1M = d1M.apply(pd.to_numeric, errors='coerce')
d1M_no = d1M[~(d1M > 325).any(axis=1)]

# Looking at the revised dataset with another scatterplot.

sns.scatterplot(d1M_no, x='Weight', y='Chest', alpha=0.5)
plt.show()

# Now to start the regression testing by looking at correlation between the variables.

print(d1M_no['Weight'].corr(d1M_no['Chest']).round(2))

# With a correlation of .87, I can say with confidence that they are eligible for regression. So, separate them:

y = d1M_no['Chest']
X = d1M_no['Weight']

# A constant is added for the intercept.

import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
X = sm.add_constant(X)
X.head()

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# The regression resulted in a Weight coefficient of .0905 and a constant coefficient/intercept of 25.5242.

# R^2 is used due to the size of the dataset. R^2 is .756. Using Weight values to predict Chest values is somewhat reliable, but could be better predicted by a different/more variables.

ax = sns.regplot(x="Weight", y="Chest", data=d1M_no)
plt.show()

# STDEV of the squared residuals is 2.12. This implies that the data values are mostly clustered around the regression line, which can be seen with the scatterplot.
