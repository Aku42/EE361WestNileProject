import numpy as np
import csv
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import LinearRegression
import matplotlib as plt
import seaborn as sns
import math

# Load dataset
X_train = pd.read_csv('input/train.csv')
X_test = pd.read_csv('input/test.csv')
sample = pd.read_csv('input/sampleSubmission.csv')
weather = pd.read_csv('input/weather.csv')
spray = pd.read_csv('input/spray.csv')

y_train = X_train['WnvPresent']


X_true = X_train[X_train.WnvPresent == 1]
plot = sns.regplot('Longitude', 'Latitude', X_true, fit_reg=False)
plot.set_ylim(41.6,42.05)
plot.set_xlim(-87.95,-87.5)
fig = plot.get_figure()
fig.savefig('scatter1.png')


