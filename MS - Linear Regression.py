import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# For this assignment, I used a dataset containing body measurements of customers for an online clothing retailer.

d1 = pd.read_csv('C:/Users/skyef/Documents/homework/data/BodyMeas.csv')
print(d1.head())
d1.info()

# What is the gender split?

d1.Gender.value_counts()

# We will be looking exclusvely at male customers.

d1M = d1[d1['Gender'].str.contains('M')]
