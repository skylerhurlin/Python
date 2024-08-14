import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# For this analysis, I used a dataset similar to the one I used for linear regression, just larger. I will be using a variety of variables to predict Weight.

d1 = pd.read_excel('data/BodyMeas2500.xlsx')

print(d1.shape)

print(d1.isna().sum())
print('\nTotal Missing:', d1.isna().sum().sum())

# We will be using both Male and Female entries this time, and will use a dummy variable Gender_M to indicate whether an entry is Male or Female. Female is dropped to avoid redundancy.

d1d = pd.get_dummies(d1, columns=['Gender'], drop_first=True)
d1d.head()

# Next, the data is separated into features and target.

y = d1d['Weight']
preds = ['Height', 'Waist', 'Hips', 'Chest', 'ArmLength', 'Gender_M']
X = d1d[preds]
X.head()

from sklearn.linear_model import LinearRegression
reg_model = LinearRegression()

# Make a train/test split, using 75% of the data for training and 25% for testing.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=7)

reg_model.fit(X_train, y_train)

# Display the intercept and coefficients.

print(f'Intercept: {reg_model.intercept_:.3f}', '\n')
cdf = pd.DataFrame(reg_model.coef_, X.columns, columns=['Coefficients'])
print(cdf.round(3))

# Gender_M has the highest coef, followed by Chest and Height. ArmLength has the lowest.

y_fit = reg_model.predict(X_train)
y_pred = reg_model.predict(X_test)

# Examine the differences between the predicted and actual values of Weight.

sns.set(rc={"figure.figsize":(6, 4)})
ax = sns.regplot(x=y_test, y=y_pred)
ax.set(xlabel='y from testing data', ylabel='predicted value of y')
plt.show()

# The graph shows that most values cluster closely to the regression line. Important to note that there are a few significant outliers, such as an individual whose actual weight was ~475 but was predicted to be ~190.

# Now to look at the mean squared error (MSE), R^2, and range of residuals for the training data.

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_train, y_fit)
rsq = r2_score(y_train, y_fit)
print("MSE: %.3f" % mse)
se = np.sqrt(mse)
range95 = 4 * se
print("Stdev of residuals: %.3f " % se)
print("Approximate 95 per cent range of residuals: %.3f " % range95)
print("R-squared: %.3f" % rsq)

# MSE was 177.567, which may indicate poor fit. R^2 is .906, quite close to 1, and the STDEV of residuals is 13.325. The model appears to be a good fit with the training data.

# Now to compare it with the testing data.

mse_f = mean_squared_error(y_test, y_pred)
rsq_f = r2_score(y_test, y_pred)
print('Forecasting Mean squared error: %.3f' % mse_f)
print('Forecasting Standard deviation of residuals: %.3f' % np.sqrt(mse_f))
print('Forecasting R-squared: %.3f' % rsq_f)

# MSE increased significantly to 318.893. STDEV increased to 17.858. R^2 decreased slightly to .858. It is common for R^2 to be smaller for testing data, but the increase in MSE is noticable. The model does still seem to be a good fit.

# This has all been done with a single train/test split. Now I will use k-fold cross-validation to run 5 splits.

from sklearn.model_selection import KFold, cross_validate
kf = KFold(n_splits=5, shuffle=True, random_state=1)

scores = cross_validate(reg_model, X, y, cv=kf,
scoring=('r2', 'neg_mean_squared_error'),
return_train_score=True)

ds = pd.DataFrame(scores)
ds.rename(columns = {'test_neg_mean_squared_error': 'test_MSE',
'train_neg_mean_squared_error': 'train_MSE'}, inplace=True)
ds['test_MSE'] = -ds['test_MSE']
ds['train_MSE'] = -ds['train_MSE']
print(ds.round(4))

# In all but one analysis, train_MSE is larger than test_MSE, indicating the model is not overfit. R^2 for testing data is between .8309 and .9202. R^2 for training data is between .8842 and .9092. All high values that imply the model is a good fit.

print('Mean of test R-squared scores: %.3f' % ds['test_r2'].mean())
print('\n')
print('Mean of test MSE scores: %.3f' % ds['test_MSE'].mean())
se = np.sqrt(ds['test_MSE'].mean())
print('Standard deviation of mean test MSE scores: %.3f' % se)

# The mean of the test R^2 is .892, the mean of its MSE is 214.640, and the STDEV of the means are 14.651. The MSE and STDEV are between the test/train values for the single train/test split done before. I can conclude that the model is a good fit.

# Lastly, I will use sklearn's feature selection to choose from the given predictor variables to run another regression.

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Predictor'] = X.columns

X = X.astype(int)
vif['VIF'] = [variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])]

cr = d1d.corr()['Weight'].round(3)
vif['Relevance'] = [cr[i]
    for i in range(X.shape[1])]

print(vif)

# Displaying the variance inflation factor can help with noticing multicollinearity. In this case, almost all of the VIFs were very large (~300-~420) except for Gender_M, which was 6.26. This indicates a lot of multicollinearity. Relevance, which shows each variable's correlation with Weight, was between .62 and 1 for all variables except for Gender_M, which was .593.

from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
from sklearn.feature_selection import RFE
selector = RFE(estimator, n_features_to_select=3, step=1).fit(X,y)

X2 = X.iloc[:, selector.support_]
X2.head()

# Sklearn has chosen Hips, Chest, and Gender_M as the best predictor variables for Weight. This excludes Height, which had a Relevance of exactly 1, possibly because its VIF was so high.

# Now another regression model will be run, but using only the chosen 3 variables.

reg_model2 = LinearRegression()
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=.25, random_state=7)
reg_model2.fit(X2_train, y2_train)

# Display the intercept and coefficients.

print(f'Intercept: {reg_model2.intercept_:.3f}', '\n')
c2df = pd.DataFrame(reg_model2.coef_, X2.columns, columns=['Coefficients'])
print(c2df.round(3))

y2_fit = reg_model2.predict(X2_train)
y2_pred = reg_model2.predict(X2_test)

sns.set(rc={"figure.figsize":(6, 4)})
ax = sns.regplot(x=y2_test, y=y2_pred)
ax.set(xlabel='y2 from testing data', ylabel='predicted value of y2')
plt.show()

# This scatterplot heavily resembles the scatterplot from the regression using all variables, implying the three chosen do just about as good a job at predicting Weight. I did notice a few more outliers.

# Finally, look at their metrics.

from sklearn.metrics import mean_squared_error, r2_score
mse2 = mean_squared_error(y2_train, y2_fit)
rsq2 = r2_score(y2_train, y2_fit)
print("MSE: %.3f" % mse2)
se2 = np.sqrt(mse2)
range95_2 = 4 * se2
print("Stdev of residuals: %.3f " % se2)
print("Approximate 95 per cent range of residuals: %.3f " % range95_2)
print("R-squared: %.3f" % rsq2)

# Fit has gone down with fewer features when compared to the 5-fold cross-validation. MSE has increased to 279.209, STDEV of residuals has increased to 16.710, and R^2 has decreased to .852.


