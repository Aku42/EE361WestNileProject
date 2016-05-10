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

def create_month(x):
    return x.split('-')[1]

def create_day(x):
    return x.split('-')[2]


def create_week(x):
        month = int(create_month(x))
        day = int(create_day(x))
        dayarr = [31,28,31,30,31,30,31,31,30,31,30,31]
        for i in [0,month-1]:
            day+=dayarr[i]
        return day/7

def create_year(x):
    return x.split('-')[0]

# Load dataset
X_train = pd.read_csv('input/train.csv', parse_dates=['Date'])
X_test = pd.read_csv('input/test.csv')
sample = pd.read_csv('input/sampleSubmission.csv')
weather = pd.read_csv('input/weather.csv')
spray = pd.read_csv('input/spray.csv')

y_train = X_train['WnvPresent']
X_train['Year'] = X_train['Date'].map(lambda x: x.year)
X_train['Week'] = X_train['Date'].map(lambda x: x.week)

X_true = X_train[X_train.WnvPresent == 1]
plot = sns.regplot('Longitude', 'Latitude', X_true, fit_reg=False)

plotX1 = X_true['Longitude'].values
plotY1 = X_true['Latitude'].values

#plotX1 = X_true[X_true.Year == '2007']['Longitude'].values
#plotY1 = X_true[X_true.Year == '2007']['Latitude'].values


#plotX2 = X_true[X_true.Year == '2009']['Longitude'].values
#plotY2 = X_true[X_true.Year == '2009']['Latitude'].values


#plotX3 = X_true[X_true.Year == '2011']['Longitude'].values
#plotY3 = X_true[X_true.Year == '2011']['Latitude'].values


#plotX4 = X_true[X_true.Year == '2013']['Longitude'].values
#plotY4 = X_true[X_true.Year == '2013']['Latitude'].values
#plot.set_ylim(41.6,42.05)
#plot.set_xlim(-87.95,-87.5)
#fig = plot.get_figure()
#fig.savefig('scatter1.png')

# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='merc' ,llcrnrlat=41.599501,urcrnrlat=42.109914,\
            llcrnrlon=-88.034279,urcrnrlon= -87.296822,lat_ts=20,resolution='f')
m.drawcoastlines()
# draw coastlines.
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
plotX, plotY = m(plotX1,plotY1)
m.drawmapboundary(fill_color='aqua')
# fill continents, set lake color same as ocean color.
m.fillcontinents(color='lightgrey',lake_color='blue', zorder=0)
m.drawparallels(np.arange(41.5, 42.5, .1))
m.drawmeridians(np.arange(-88., -87.3, .1))
m.scatter(plotX, plotY, c='red', marker="o", alpha=.8)
fig = plot.get_figure()
fig.savefig('map.png')

print("Here")
checks = X_train[['Week', 'Year', 'WnvPresent']].groupby(['Week', 'Year']).count().reset_index()
weekly_postives = X_train.groupby(['Year', 'Week', 'Species']).sum().reset_index()
weekly_postives_species = weekly_postives.set_index(['Year', 'Week', 'Species']).unstack()
weekly_postives_species.columns = weekly_postives_species.columns.get_level_values(1)
weekly_postives_species['total_positives'] = weekly_postives_species.sum(axis=1)
weekly_postives_species = weekly_postives_species.reset_index().fillna(0)


weekly_postives_weeks = weekly_postives.set_index(['Year', 'Week']).unstack()
weekly_postives_weeks.columns = weekly_postives_species.columns.get_level_values(1)
weekly_postives_species['total_positives'] = weekly_postives_weeks.sum(axis=1)

weekly_checks = checks.groupby(['Year', 'Week']).sum()
weekly_checks.columns = ['checks']
weekly_checks = weekly_checks.reset_index()
weekly_checks['positive'] = weekly_postives_species['total_positives']
weekly_checks['trap_infection_rate'] = weekly_postives_weeks
weekly_checks_years = weekly_checks.pivot(index='Week', columns='Year', values='trap_infection_rate')

ax = weekly_checks_years.interpolate().plot(title='Trap infection rate', figsize=(10, 6))
ax.set_ylabel('Number of traps infected')
fig = ax.get_figure()
fig.savefig('positive_trap_rate.png')

