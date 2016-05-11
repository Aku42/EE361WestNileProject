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
from mpl_toolkits.basemap import Basemap
import string
import matplotlib.cm as cm
import seaborn as sns
import math



# Load dataset
X_train = pd.read_csv('input/train.csv', parse_dates=['Date'])
X_test = pd.read_csv('input/test.csv')
sample = pd.read_csv('input/sampleSubmission.csv')
weather = pd.read_csv('input/weather.csv')
spray = pd.read_csv('input/spray.csv')

y_train = X_train['WnvPresent']
X_train['Year'] = X_train['Date'].map(lambda x: x.year)
X_train['Week'] = X_train['Date'].map(lambda x: x.week)

X_true = X_train[X_train.WnvPresent >= 0]
plot = sns.regplot('Longitude', 'Latitude', X_true, fit_reg=False)

plotX = X_true['Longitude'].values
plotY = X_true['Latitude'].values
plotZ = X_true['WnvPresent'].values

plotX1 = X_true[X_true.Year == 2007]['Longitude'].values
plotY1 = X_true[X_true.Year == 2007]['Latitude'].values


plotX2 = X_true[X_true.Year == 2009]['Longitude'].values
plotY2 = X_true[X_true.Year == 2009]['Latitude'].values


plotX3 = X_true[X_true.Year == 2011]['Longitude'].values
plotY3 = X_true[X_true.Year == 2011]['Latitude'].values

plotX4 = X_true[X_true.Year == 2013]['Longitude'].values
plotY4 = X_true[X_true.Year == 2013]['Latitude'].values


# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='merc', llcrnrlat=41.599501, urcrnrlat=42.109914, llcrnrlon=-88.034279, urcrnrlon=-87.296822,
            lat_ts=20, resolution='f')
m.drawcoastlines()
# draw coastlines.
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
plotX1, plotY1 = m(plotX1, plotY1)
plotX2, plotY2 = m(plotX2, plotY2)
plotX3, plotY3 = m(plotX3, plotY3)
plotX4, plotY4 = m(plotX4, plotY4)
m.drawmapboundary(fill_color='blue')
# fill continents, set lake color same as ocean color.
m.fillcontinents(color='lightgrey',lake_color='blue', zorder=0)
m.drawparallels(np.arange(41.5, 42.5, .1))
m.drawmeridians(np.arange(-88., -87.3, .1))
m.scatter(plotX1, plotY1, c='red', marker="o", alpha=1)
m.scatter(plotX2, plotY2, c='purple', marker="o", alpha=1)
m.scatter(plotX3, plotY3, c='green', marker="o", alpha=1)
m.scatter(plotX4, plotY4, c='orange', marker="o", alpha=1)
fig = plot.get_figure()
fig.savefig('map.png')

print("Here")

# m = Basemap(projection='merc', llcrnrlat=41.599501, urcrnrlat=42.109914, llcrnrlon=-88.034279, urcrnrlon=-87.296822,
#             lat_ts=20, resolution='f')
# m.drawcoastlines()
# # draw coastlines.
# # draw a boundary around the map, fill the background.
# # this background will end up being the ocean color, since
# m.drawparallels(np.arange(41.5, 42.5, .1))
# m.drawmeridians(np.arange(-88., -87.3, .1))
# plotX, plotY = m(plotX, plotY)
# m.pcolormesh(plotX, plotY, plotZ, interpolation='none', cmap='Oranges')
# fig = plot.get_figure()
# fig.savefig('heatmap.png')
#
# print("Here")

X_2007 = X_true[X_true.Year == 2007]
X_2009 = X_true[X_true.Year == 2009]
X_2011 = X_true[X_true.Year == 2011]
X_2013 = X_true[X_true.Year == 2013]

weekly = {}
weekly['2007'] = X_2007.groupby("Week").sum().ix[:,['WnvPresent']]
weekly['2009'] = X_2009.groupby("Week").sum().ix[:,['WnvPresent']]
weekly['2011'] = X_2011.groupby("Week").sum().ix[:,['WnvPresent']]
weekly['2013'] = X_2013.groupby("Week").sum().ix[:,['WnvPresent']]
for i in weekly:
    weekly[i].rename(columns={'WnvPresent': i}, inplace=True)
ax = weekly['2007'].plot.line(title='Trap infection rate', figsize=(10, 6),c="red")
ax = weekly['2009'].plot.line(ax=ax)
ax = weekly['2011'].plot.line(ax=ax)
ax = weekly['2013'].plot.line(ax=ax, c="orange")
ax.set_ylabel('Traps infected')
ax.get_figure().savefig('positive_trap_rate.png')
