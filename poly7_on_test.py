"""

  Created on 6/19/2017 by Ben

  benuklove@gmail.com
  
  Use seventh degree polynomial regression on full_sq for test data

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


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

df_train = pd.read_csv("../data/train.csv")

# Suppress SettingWithCopyWarning
# default='warn'
pd.options.mode.chained_assignment = None

# Remove outliers
ulimit = np.percentile(df_train.price_doc.values, 99.5)
llimit = np.percentile(df_train.price_doc.values, 0.5)
df_train['price_doc'].loc[df_train['price_doc']>ulimit] = ulimit
df_train['price_doc'].loc[df_train['price_doc']<llimit] = llimit

col = "full_sq"
ulimit = np.percentile(df_train[col].values, 99.5)
llimit = np.percentile(df_train[col].values, 0.5)
df_train[col].loc[df_train[col]>ulimit] = ulimit
df_train[col].loc[df_train[col]<llimit] = llimit

# For plotting purposes, sort by 'full_sq'
df_train.sort_values(['full_sq', 'price_doc'], inplace=True)

# Degree seven polynomial, on all training data
poly7_data = polynomial_df(df_train['full_sq'], 7)
my_features = poly7_data.columns.values

# Add target 'price' to the new DataFrame
poly7_data['price'] = df_train['price_doc']

# Convert pandas Series to numpy array
xs = poly7_data[my_features].values
ys = poly7_data['price'].values

# Create linear regression object and train the model
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(xs, ys)

# Read in test data
df_test = pd.read_csv("../data/test.csv", usecols=["id", "full_sq"])

# Organize shape of test data similar to training model
test_data = polynomial_df(df_test['full_sq'], 7)
test_features = test_data.columns.values

# Identify test feature as numpy array
xs_test = test_data[test_features].values

# Predict price with trained model
test_out = regr.predict(xs_test)

# Format output
test_out = pd.Series(test_out).round(decimals=2)
results = pd.concat([df_test['id'], test_out], axis=1)
results.columns = ['id', 'price_doc']

# Send to submission file
results.to_csv("../Submissions/predictions.csv", index=False, encoding="utf-8")
