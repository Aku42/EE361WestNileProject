import numpy as np
import csv
import itertools as ite
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import LinearRegression
import matplotlib as plt
from mpl_toolkits.basemap import Basemap
import string
import matplotlib.cm as cm
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
plotX = X_true['Longitude'].values
plotY = X_true['Latitude'].values
#plot.set_ylim(41.6,42.05)
#plot.set_xlim(-87.95,-87.5)
#fig = plot.get_figure()
#fig.savefig('scatter1.png')

#drawing the map underneath
m = Basemap(projection='merc' ,llcrnrlat=41.599501,urcrnrlat=42.109914, llcrnrlon=-88.034279,urcrnrlon= -87.296822,lat_ts=20,resolution='f')
m.drawcoastlines()
plotX, plotY = m(plotX,plotY)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='lightgrey', lake_color='blue', zorder=0)
m.scatter(plotX, plotY, c='red', marker="o", alpha=.8)
fig = plot.get_figure()
fig.savefig('map.png')


X_test["week"]