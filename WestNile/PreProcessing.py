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
import math, copy, datetime

def preProcessData():
    # Load dataset
    X_train = pd.read_csv('input/train.csv', parse_dates="Date")
    X_test = pd.read_csv('input/test.csv', parse_dates="Date")
    sample = pd.read_csv('input/sampleSubmission.csv')
    weather = pd.read_csv('input/weather.csv', parse_dates="Date")
    spray = pd.read_csv('input/spray.csv', parse_dates="Date")

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

    def create_year(x):
        return x.split('-')[0]

    #X_train['month'] = X_train.Date.apply(create_month)
    #X_train['day'] = X_train.Date.apply(create_day)
    #X_train['week'] = X_train.Date.apply(create_week)
    #X_train['year'] = X_train.Date.apply(create_year)
    #X_test['month'] = X_test.Date.apply(create_month)
    #X_test['day'] = X_test.Date.apply(create_day)
    #X_test['week'] = X_test.Date.apply(create_week)
    #X_test['year'] = X_train.Date.apply(create_year)

    print X_train.head(5)

    X_true = X_train[X_train.WnvPresent == 1]
    plot = sns.regplot('Longitude', 'Latitude', X_true, fit_reg=False)
    plot.set_ylim(41.6,42.05)
    plot.set_xlim(-87.95,-87.5)
    fig = plot.get_figure()
    fig.savefig('scatter1.png')

    # drop address columns
    X_train = X_train.drop(['Address', 'AddressNumberAndStreet', 'WnvPresent', 'Street', 'NumMosquitos', 'Trap'], axis = 1)
    X_test = X_test.drop(['Id', 'Address', 'AddressNumberAndStreet', 'Street', 'Trap'], axis = 1)

    # Convert categorical data to numbers
    lbl = LabelEncoder()
    lbl.fit(list(X_train['Species'].values) + list(X_test['Species'].values))
    X_train['Species'] = lbl.transform(X_train['Species'].values)
    X_test['Species'] = lbl.transform(X_test['Species'].values)

    #lbl.fit(list(X_train['Street'].values) + list(X_test['Street'].values))
    #X_train['Street'] = lbl.transform(X_train['Street'].values)
    #X_test['Street'] = lbl.transform(X_test['Street'].values)

    #lbl.fit(list(X_train['Trap'].values) + list(X_test['Trap'].values))
    #X_train['Trap'] = lbl.transform(X_train['Trap'].values)
    #X_test['Trap'] = lbl.transform(X_test['Trap'].values)

    # drop columns with -1s
    X_train = X_train.ix[:, (X_train != -1).any(axis=0)]
    X_test = X_test.ix[:, (X_test != -1).any(axis=0)]


    # Merge with weather data
    X_train = X_train.merge(weather, on='Date')
    X_test = X_test.merge(weather, on='Date')
    X_train = X_train.drop(['Date'], axis = 1)
    X_test = X_test.drop(['Date'], axis = 1)

    return X_train, y_train, X_test, sample

def preProcessData_MergeClosest():
    # Load dataset
    X_train = pd.read_csv('input/train.csv', parse_dates="Date")
    X_test = pd.read_csv('input/test.csv', parse_dates="Date")
    sample = pd.read_csv('input/sampleSubmission.csv')
    weather = pd.read_csv('input/weather.csv', parse_dates="Date")
    spray = pd.read_csv('input/spray.csv', parse_dates="Date")

    y_train = X_train['WnvPresent']

    weather = weather.drop('CodeSum', axis=1)

    # replace some missing values and T with -1
    weather = weather.replace('M', -1)
    weather = weather.replace('-', -1)
    weather = weather.replace('T', -1)
    weather = weather.replace(' T', -1)
    weather = weather.replace('  T', -1)

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

    X_train['Id'] = range(0, len(X_train))

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
        return (day/7)%1

    def create_year(x):
        return x.split('-')[0]

    # X_train['month'] = X_train.Date.apply(create_month)
    # X_train['day'] = X_train.Date.apply(create_day)
    # X_train['week'] = X_train.Date.apply(create_week)
    # X_train['year'] = X_train.Date.apply(create_year)
    # X_test['month'] = X_test.Date.apply(create_month)
    # X_test['day'] = X_test.Date.apply(create_day)
    # X_test['week'] = X_test.Date.apply(create_week)
    # X_test['year'] = X_train.Date.apply(create_year)

    # drop address columns
    X_train = X_train.drop(['Address', 'AddressNumberAndStreet', 'WnvPresent', 'Street', 'NumMosquitos', 'Trap'], axis = 1)
    X_test = X_test.drop(['Address', 'AddressNumberAndStreet', 'Street', 'Trap'], axis = 1)

    # Convert categorical data to numbers
    lbl = LabelEncoder()
    lbl.fit(list(X_train['Species'].values) + list(X_test['Species'].values))
    X_train['Species'] = lbl.transform(X_train['Species'].values)
    X_test['Species'] = lbl.transform(X_test['Species'].values)

    #lbl.fit(list(X_train['Street'].values) + list(X_test['Street'].values))
    #X_train['Street'] = lbl.transform(X_train['Street'].values)
    #X_test['Street'] = lbl.transform(X_test['Street'].values)

    #lbl.fit(list(X_train['Trap'].values) + list(X_test['Trap'].values))
    #X_train['Trap'] = lbl.transform(X_train['Trap'].values)
    #X_test['Trap'] = lbl.transform(X_test['Trap'].values)

    # drop columns with -1s
    X_train = X_train.ix[:, (X_train != -1).any(axis=0)]
    X_test = X_test.ix[:, (X_test != -1).any(axis=0)]


    def merge_closest(DF1, DF2, DF3, on_col, x_col, y_col):
        #matrix1 = DF1.values
        #matrix2 = DF2.values
        tempDF = DF2.copy()
        i=0
        startTime = datetime.datetime.now()
        print "Merging..."
        print "NumRows = %s" % len(DF1)
        for index,row in DF1.iterrows():
            df1_date = row[on_col]
            df2_row = DF2[(DF2[on_col]==df1_date)]
            df3_row = DF3[(DF3[on_col]==df1_date)]
            df1_x=row[x_col]
            df1_y=row[y_col]
            df2_x=df2_row[x_col]
            df2_y=df2_row[y_col]
            df3_x=df3_row[x_col]
            df3_y=df3_row[y_col]
            df2_dist=math.hypot(df2_x-df1_x, df2_y-df1_y)
            df3_dist=math.hypot(df3_x-df1_x, df3_y-df1_y)
            if i%500==0:
                print "At row %s" % i
            if df2_dist<=df3_dist:
                tempDF[tempDF[on_col]==df1_date] = np.array(df2_row)
            else:
                tempDF[tempDF[on_col]==df1_date] = np.array(df3_row)
            i+=1
        print "Done! Elapsed Time: " + str(datetime.datetime.now()-startTime)
        return tempDF


    # Merge with weather data
    mergedWeather = merge_closest(X_train, weather_stn1, weather_stn2,on_col='Date', x_col='Latitude', y_col='Longitude')
    X_train = X_train.merge(mergedWeather, on='Date')
    print X_train.head(5)
    mergedWeather = merge_closest(X_test, weather_stn1, weather_stn2,on_col='Date', x_col='Latitude', y_col='Longitude')
    X_test = X_test.merge(mergedWeather, on='Date')
    print X_test.head(5)
    X_train = X_train.drop(['Date'], axis = 1)
    X_test = X_test.drop(['Date'], axis = 1)

    return X_train, y_train, X_test, sample



if __name__ == '__main__':
    #X_train, y_train, X_test, sample = preProcessData_MergeClosest()
    #X_train = X_train.drop(['Latitude_y','Longitude_y'], axis = 1)
    #X_train.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'}, inplace=True)
    #X_test.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'}, inplace=True)
    #X_test= X_test.drop(['Latitude_y','Longitude_y'], axis = 1)
    #X_train.to_csv('predata/trainmerged.csv', index=False)
    #X_test.to_csv('predata/testmerged.csv', index=False)
    X_train = pd.read_csv('predata/trainmerged.csv', parse_dates="Date")
    X_test = pd.read_csv('predata/testmerged.csv', parse_dates="Date")
    sample = pd.read_csv('input/sampleSubmission.csv')
    y_train =pd.read_csv('input/train.csv', parse_dates="Date")['WnvPresent']
    krig_train = X_train.ix[:,['Id','Latitude','Longitude']]
    krig_test = X_test.ix[:,['Id','Latitude','Longitude']]
    log_test = X_test.drop(['Latitude','Longitude'],axis=1)
    log_train = X_train.drop(['Latitude','Longitude'],axis=1)
    gp = LinearRegression(theta0=1e-2)
    gp.fit(log_train, y_train)
    train_pred = gp.predict(X=log_train)
    temp_pred = gp.predict(X=log_test)
    krig_train['logpred']=train_pred
    krig_test['logpred']=temp_pred

    gp = GaussianProcess(theta0=1e12)
    gp.fit(krig_train, y_train)
    predictions = gp.predict(krig_test)

    #predictions = gp.predict(X_test)
    sample['WnvPresent'] = predictions
    sample.to_csv('firstGP.csv', index=False)

