# This is a segment of a homework assignment for my course Machine Learning for Business.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1: Evaluating a dataset of bank client information.

d1 = pd.read_csv("C:/Users/skyef/Documents/homework/data/bank-full.csv")
print(d1.head(10))
d1.info()

# What's the split of married, single, and divorced?

d1.marital.value_counts()
d1['marital'] = pd.Categorical(d1['marital'], categories=['single', 'married', 'divorced'], ordered=True)

sns.countplot(x='marital', data=d1, hue='marital')
plt.show()

# Comparing histograms with different bins - which is better?

sns.histplot(d1, x="age")
plt.show()

sns.histplot(d1, x="age", binwidth=4)
plt.show()

# Changing the default bin width smooths out the dataset and makes the trend more obvious.

# Part 2: Boston house prices and related features.

d2 = pd.read_csv("C:/Users/skyef/Documents/homework/data/boston.csv")
d2.head(5)
d2.info()

# Changing a percentage variable into a proportion.

d2['lstatProp'] = d2['lstat']/100
d2.head(5)

# What is the average number of rooms (rm) for the second row of data? Use iloc.

print(d2.iloc[1, 6])

# We will focus on three predictors: per capita crime rate (crim), average no. of rooms (rm), and index of accessibility to radial highways (rad).

print(d2.loc[:, ['crim','rm','rad']].head(5))
# or
print(d2.iloc[:, [1,5,8]].head(5))

# List all the rows of data with the median value of the home less than 8000.

print(d2.query('medv < 8'))

# Construct a box plot of the target variable medv.

sns.set(rc={"figure.figsize":(8,1.5)})
ax = sns.boxplot(x=d2['medv'], color='steelblue')
ax.set(xlabel='Median Home Value (in 1000s USD)', ylabel='')
plt.show()

# Lastly, experiment with scaling methods. Use the MinMax scaler to convert the three features so all values are between 0 and 1.

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
mm_scaler = preprocessing.MinMaxScaler()

X = d2[['crim','rm','rad']].copy()
X.head(5)

Xmm = mm_scaler.fit_transform(X)
Xmm = pd.DataFrame(Xmm, columns=['crim', 'rm', 'rad'])
Xmm.head(5)

# Test the max and min to confirm they are 1 and 0 respectively.

print(Xmm.min())
print(Xmm.max())

# Now use the Standard Scaler.

from sklearn.preprocessing import StandardScaler
s_scaler = preprocessing.StandardScaler()

Xst = s_scaler.fit_transform(X)
Xst = pd.DataFrame(Xst, columns=['crim', 'rm', 'rad'])
Xst.head(5)

# Test the mean and STDEV.

round(Xst.mean(), 4)
round(Xst.std(), 4)

# The -0.0 for the mean implies that the mean for all three standardized variables is negative and very, very close to zero. 1.001 for the standard deviations means that the data points are largely clustered around the mean and there is not a lot of variability between the values of all three variables.

# Use the Robust Scaler.

from sklearn.preprocessing import RobustScaler
r_scaler = preprocessing.RobustScaler()

Xrb = r_scaler.fit_transform(X)
Xrb = pd.DataFrame(Xrb, columns=['crim', 'rm', 'rad'])
Xrb.head(5)

# Part 3: Supermarket transactions.

d3 = pd.read_excel("C:/Users/skyef/Documents/homework/data/SupermarketTransactions.xlsx")

# Practice with making dummy variables. What are the three countries in the table?

print(d2['Country'].unique())
d2d = pd.get_dummies(d2, columns=['Country'], drop_first=True)
d2d.head()

# One of the three countries can be dropped making dummies because its value can be implied by the other two dummies.
