import numpy as np
import csv
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt
import seaborn as sns


# Load dataset
X_train = pd.read_csv('input/train.csv')
X_test = pd.read_csv('input/test.csv')
sample = pd.read_csv('input/sampleSubmission.csv')
weather = pd.read_csv('input/weather.csv')

# Get labels
labels = X_train.WnvPresent.values

# Not using codesum for this benchmark
weather = weather.drop('CodeSum', axis=1)

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
weather = weather.replace(' T', -1)
weather = weather.replace('  T', -1)

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]

X_train['month'] = X_train.Date.apply(create_month)
X_train['day'] = X_train.Date.apply(create_day)
X_test['month'] = X_test.Date.apply(create_month)
X_test['day'] = X_test.Date.apply(create_day)

# Add integer latitude/longitude columns
X_train['Lat_int'] = X_train.Latitude.apply(int)
X_train['Long_int'] = X_train.Longitude.apply(int)
X_test['Lat_int'] = X_test.Latitude.apply(int)
X_test['Long_int'] = X_test.Longitude.apply(int)

X_true = X_train[X_train.WnvPresent == 1]
plot = sns.regplot('Longitude', 'Latitude', X_true, fit_reg=False)
fig = plot.get_figure()
fig.savefig('scatter1.png')

# drop address columns
X_train = X_train.drop(['Address', 'AddressNumberAndStreet', 'WnvPresent', 'NumMosquitos'], axis = 1)
X_test = X_test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

# Merge with weather data
X_train = X_train.merge(weather, on='Date')
X_test = X_test.merge(weather, on='Date')
X_train = X_train.drop(['Date'], axis = 1)
X_test = X_test.drop(['Date'], axis = 1)
