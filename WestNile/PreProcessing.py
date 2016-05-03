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

def preProcessData():
    # Load dataset
    X_train = pd.read_csv('input/train.csv')
    X_test = pd.read_csv('input/test.csv')
    sample = pd.read_csv('input/sampleSubmission.csv')
    weather = pd.read_csv('input/weather.csv')
    spray = pd.read_csv('input/spray.csv')

    y_train = X_train['WnvPresent']

    weather = weather.drop('CodeSum', axis=1)

    # Split station 1 and 2 and join horizontally
    weather_stn1 = weather[weather['Station']==1]
    weather_stn2 = weather[weather['Station']==2]
    #weather_stn1 = weather_stn1.drop('Station', axis=1)
    weather_stn1.loc[:,'Latitude'] = 41.995
    weather_stn1.loc[:,'Longitude'] = -87.933
    weather_stn2.loc[:,'Latitude'] = 41.786
    weather_stn2.loc[:,'Longitude'] = -87.752
    #weather_stn2 = weather_stn2.drop('Station', axis=1)
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

    def create_week(x):
        month = int(create_month(x))
        day = int(create_day(x))
        dayarr = [31,28,31,30,31,30,31,31,30,31,30,31]
        for i in [0,month-1]:
            day+=dayarr[i]
        return day/7

    X_train['month'] = X_train.Date.apply(create_month)
    X_train['day'] = X_train.Date.apply(create_day)
    X_train['week'] = X_train.Date.apply(create_week)
    X_test['month'] = X_test.Date.apply(create_month)
    X_test['day'] = X_test.Date.apply(create_day)
    X_test['week'] = X_test.Date.apply(create_week)

    X_true = X_train[X_train.WnvPresent == 1]
    plot = sns.regplot('Longitude', 'Latitude', X_true, fit_reg=False)
    plot.set_ylim(41.6,42.05)
    plot.set_xlim(-87.95,-87.5)
    fig = plot.get_figure()
    fig.savefig('scatter1.png')

    # drop address columns
    X_train = X_train.drop(['Address', 'AddressNumberAndStreet', 'WnvPresent', 'Street', 'NumMosquitos', 'Trap'], axis = 1)
    X_test = X_test.drop(['Id', 'Address', 'AddressNumberAndStreet', 'Street', 'Trap'], axis = 1)

    #speciesNames = {}
    #x = 0
    #for i in X_train.Species.values:
    #    if i not in speciesNames:
    #        speciesNames[i]=2*x
    #        print i
    #        x+=1

    # replace mosquito species with int
    #X_train['Species'] = X_train['Species'].map(speciesNames)
    #X_test['Species'] = X_test['Species'].map(speciesNames)
    X_train = X_train.replace('CULEX RESTUANS',2)
    X_train = X_train.replace('CULEX PIPIENS/RESTUANS',3)
    X_train = X_train.replace('CULEX PIPIENS',4)
    X_train = X_train.replace('CULEX SALINARIUS',6)
    X_train = X_train.replace('CULEX TERRITANS',8)
    X_train = X_train.replace('CULEX TARSALIS',10)
    X_train = X_train.replace('CULEX ERRATICUS',12)
    X_train = X_train.replace('UNSPECIFIED CULEX',0)

    X_test = X_test.replace('CULEX RESTUANS',2)
    X_test = X_test.replace('CULEX PIPIENS/RESTUANS',3)
    X_test = X_test.replace('CULEX PIPIENS',4)
    X_test = X_test.replace('CULEX SALINARIUS',6)
    X_test = X_test.replace('CULEX TERRITANS',8)
    X_test = X_test.replace('CULEX TARSALIS',10)
    X_test = X_test.replace('CULEX ERRATICUS',12)
    X_test = X_test.replace('UNSPECIFIED CULEX',0)


    def merge_closest(DF1, DF2, div_col, div2_val, div3_val, on_col, x_col, y_col):
        #matrix1 = DF1.values
        #matrix2 = DF2.values
        tempDF = pd.DataFrame(data=None, columns=DF2.columns, index=DF2.index)
        i=0
        for index,row in DF1.iterrows():
            df1_date = row[on_col]
            df2_row = DF2[(DF2[on_col]==df1_date) & (DF2[div_col]==div2_val)]
            df3_row = DF2[(DF2[on_col]==df1_date) & (DF2[div_col]==div3_val)]
            print df2_row
            df1_x=row[x_col]
            df1_y=row[y_col]
            df2_x=df2_row[x_col]
            df2_y=df2_row[y_col]
            df3_x=df3_row[x_col]
            df3_y=df3_row[y_col]
            df2_dist=math.hypot(df2_x-df1_x, df2_y-df1_y)
            df3_dist=math.hypot(df3_x-df1_x, df3_y-df1_y)
            #if i<5:
            #    print "DF2: " + str(df2_dist) + " DF3: " +str(df3_dist)
            #    i+=1
            #print type(df2_row)
            if df2_dist<=df3_dist:
                tempDF=tempDF[DF2[on_col]!=df1_date & DF2[div_col]!=div3_val]
            else:
                tempDF.loc[i]=DF2[on_col]!=df1_date & DF2[div_col]!=div2_val
            i+=1
        print tempDF.head(5)


    # Merge with weather data
    #merge_closest(X_train, weather, div_col='Station', div2_val=1, div3_val=2,on_col='Date', x_col='Latitude', y_col='Longitude')
    X_train = X_train.merge(weather, on='Date')
    X_test = X_test.merge(weather, on='Date')
    X_train = X_train.drop(['Date'], axis = 1)
    X_test = X_test.drop(['Date'], axis = 1)

    return X_train, y_train, X_test, sample

X_train, y_train, X_test, sample = preProcessData()
gp = GaussianProcess()
gp.fit(X_train, y_train)
y_pred = gp.predict(X_test)

predictions = gp.predict(X_test)[:,1]
sample['WnvPresent'] = predictions
sample.to_csv('firstGP.csv', index=False)

