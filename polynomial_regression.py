"""

  Created on 6/16/2017 by Ben

  benuklove@gmail.com
  
  Polynomial Regression

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from math import sqrt

def polynomial_df(feature, degree):
    """ Takes in a pandas Series and returns a Dataframe of feature powers
    up to a specified degree. """
    poly_df = pd.DataFrame()
    poly_df['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_df[name] = feature.apply(lambda x: x ** power)
    return poly_df

def rmsle(num_observations, predictions, actual_value):
    """ Return the Root Mean Squared Log Error.
    This penalizes lower predictions more than higher ones. """
    epsilon = sqrt((sum((np.log(predictions + 1) -
                         np.log(actual_value + 1)) ** 2)) / num_observations)
    return epsilon

df = pd.read_csv("../data/train.csv")

# Suppress SettingWithCopyWarning
# default='warn'
pd.options.mode.chained_assignment = None

# Remove outliers
ulimit = np.percentile(df.price_doc.values, 99.5)
llimit = np.percentile(df.price_doc.values, 0.5)
df['price_doc'].loc[df['price_doc']>ulimit] = ulimit
df['price_doc'].loc[df['price_doc']<llimit] = llimit

col = "full_sq"
ulimit = np.percentile(df[col].values, 99.5)
llimit = np.percentile(df[col].values, 0.5)
df[col].loc[df[col]>ulimit] = ulimit
df[col].loc[df[col]<llimit] = llimit

# For plotting purposes, sort by 'full_sq'
df.sort_values(['full_sq', 'price_doc'], inplace=True)

# Degree one polynomial for now (straight line)
poly1_data = polynomial_df(df['full_sq'], 1)

# Add target 'price' to the new DataFrame
poly1_data['price'] = df['price_doc']

# Convert pandas Series to numpy array and reshape
xs = poly1_data.power_1.values.reshape(len(poly1_data['power_1']), 1)
ys = poly1_data.price.values

# Create linear regression object and train the model
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(xs, ys)

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print("Mean squared error: %.2f" % np.mean((regr.predict(xs) - ys) ** 2))
print('Variance score: %.2f' % regr.score(xs, ys))

plt.scatter(xs, ys,  color='black')
plt.plot(xs, regr.predict(xs), color='blue', linewidth=3)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Full square meters', fontsize=12)
plt.show()
print(rmsle(num_observations=len(xs),
            predictions=regr.predict(xs),
            actual_value=ys))


""" Now let's try it with higher order polynomials. """

# Degree two polynomial
poly2_data = polynomial_df(df['full_sq'], 2)

# Add target 'price' to the new DataFrame
poly2_data['price'] = df['price_doc']

# Convert pandas Series to numpy array
xs = poly2_data[['power_1', 'power_2']].values
ys = poly2_data.price.values

# Create linear regression object and train the model
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(xs, ys)

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print("Mean squared error: %.2f" % np.mean((regr.predict(xs) - ys) ** 2))
print('Variance score: %.2f' % regr.score(xs, ys))

plt.scatter(poly2_data['power_1'], ys,  color='black')
plt.plot(poly2_data['power_1'], regr.predict(xs), color='blue', linewidth=3)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Full square meters', fontsize=12)
plt.axis(xmin=20)
plt.show()
print(rmsle(num_observations=len(xs),
            predictions=regr.predict(xs),
            actual_value=ys))

# Now degree three polynomial
poly3_data = polynomial_df(df['full_sq'], 3)

# Add target 'price' to the new DataFrame
poly3_data['price'] = df['price_doc']

# Convert pandas Series to numpy array
xs = poly3_data[['power_1', 'power_2', 'power_3']].values
ys = poly3_data.price.values

# Create linear regression object and train the model
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(xs, ys)

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print("Mean squared error: %.2f" % np.mean((regr.predict(xs) - ys) ** 2))
print('Variance score: %.2f' % regr.score(xs, ys))

plt.scatter(poly3_data['power_1'], ys,  color='black')
plt.plot(poly3_data['power_1'], regr.predict(xs), color='blue', linewidth=3)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Full square meters', fontsize=12)
plt.axis(xmin=20)
plt.show()
print(rmsle(num_observations=len(xs),
            predictions=regr.predict(xs),
            actual_value=ys))

# Now degree seven polynomial, on all data
new_poly7_data = polynomial_df(df['full_sq'], 7)
my_features = new_poly7_data.columns.values

# Add target 'price' to the new DataFrame
new_poly7_data['price'] = df['price_doc']

# Convert pandas Series to numpy array
xs = new_poly7_data[my_features].values
ys = new_poly7_data.price.values

# Create linear regression object and train the model
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(xs, ys)

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print("Mean squared error: %.2f" % np.mean((regr.predict(xs) - ys) ** 2))
print('Variance score: %.2f' % regr.score(xs, ys))

plt.scatter(new_poly7_data['power_1'], ys,  color='black')
plt.plot(new_poly7_data['power_1'], regr.predict(xs), color='blue', linewidth=3)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Full square meters', fontsize=12)
plt.axis(xmin=20)
plt.show()
print(rmsle(num_observations=len(xs),
            predictions=regr.predict(xs),
            actual_value=ys))


""" We need to choose the best degree polynomial - validation. """

# Split data into training, validation, and testing data
x_train_and_valid, x_test, y_train_and_valid, y_test = train_test_split(
    df['full_sq'].values, df['price_doc'].values,
    test_size=0.1, random_state=12)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_and_valid[:], y_train_and_valid[:],
    test_size=0.5, random_state=12)

x_test = pd.Series(x_test)
y_test = pd.Series(y_test)
x_train = pd.Series(x_train)
x_valid = pd.Series(x_valid)
y_train = pd.Series(y_train)
y_valid = pd.Series(y_valid)


def rss_validate(x_train, y_train, x_valid, y_valid, highest_degree):
    """Takes in training data, validation data and the highest acceptable
    degree polynomial fit and returns a dict of RSS. """
    from operator import sub
    rss_list = {}
    for degree in range(1, highest_degree+1):
        if degree == 1:
            poly_data = polynomial_df(x_train, degree)
            poly_data['price'] = y_train
            xs = poly_data.power_1.values.reshape(len(poly_data['power_1']), 1)
            ys = poly_data.price.values
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(xs, ys)
            val_results = regr.predict(x_valid.values.reshape(len(x_valid), 1))
        else:
            poly_data = polynomial_df(x_train, degree)
            my_features = poly_data.columns.values
            poly_data['price'] = y_train
            xs = poly_data[my_features].values
            ys = y_train.values
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(xs, ys)
            val_data = polynomial_df(x_valid, degree)
            val_features = val_data.columns.values
            xs_val = val_data[val_features].values
            val_results = regr.predict(xs_val)
        diff = map(sub, val_results, y_valid)
        RSS = sum(map(lambda x: x ** 2, diff))
        rss_list[degree] = RSS
    return rss_list

rss_valid_results = rss_validate(x_train, y_train, x_valid, y_valid, 15)
print(min(rss_valid_results, key=rss_valid_results.get))


def rmsle_validate(x_train, y_train, x_valid, y_valid, highest_degree):
    """Takes in training data, validation data and the highest acceptable
    degree polynomial fit and returns a dict of RSMLE. """
    rmsle_list = {}
    for degree in range(1, highest_degree + 1):
        if degree == 1:
            poly_data = polynomial_df(x_train, degree)
            poly_data['price'] = y_train
            xs = poly_data.power_1.values.reshape(len(poly_data['power_1']), 1)
            ys = poly_data.price.values
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(xs, ys)
            val_results = regr.predict(x_valid.values.reshape(len(x_valid), 1))
        else:
            poly_data = polynomial_df(x_train, degree)
            my_features = poly_data.columns.values
            poly_data['price'] = y_train
            xs = poly_data[my_features].values
            ys = y_train.values
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(xs, ys)
            val_data = polynomial_df(x_valid, degree)
            val_features = val_data.columns.values
            xs_val = val_data[val_features].values
            val_results = regr.predict(xs_val)
        rmsle_list[degree] = rmsle(num_observations=len(val_results),
                                   predictions=val_results,
                                   actual_value=y_valid)
    return rmsle_list

rmsle_valid_results = rmsle_validate(x_train, y_train, x_valid, y_valid, 15)
print(rmsle_valid_results)
print(min(rmsle_valid_results, key=rmsle_valid_results.get))

poly7_data = polynomial_df(x_test, 7)
my_features = poly7_data.columns.values
poly7_data['price'] = y_test
poly7_data.sort_values(['power_1', 'price'], inplace=True)
xs = poly7_data[my_features].values
ys = poly7_data['price'].values
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(xs, ys)
plt.scatter(poly7_data['power_1'], ys,  color='black')
plt.plot(poly7_data['power_1'], regr.predict(xs), color='blue', linewidth=3)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Full square meters', fontsize=12)
plt.axis(xmin=20)
plt.show()

print(rmsle(num_observations=len(xs),
            predictions=regr.predict(xs),
            actual_value=ys))
