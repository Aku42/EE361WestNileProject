import numpy as np
import csv
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomTreesEmbedding, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn import ensemble, preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import grid_search
from sklearn.ensemble import AdaBoostClassifier
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
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix

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

    #def create_year(x):
     #   return x.split('-')[0]

    X_train['month'] = X_train.Date.apply(create_month)
    X_train['day'] = X_train.Date.apply(create_day)
    X_train['week'] = X_train.Date.apply(create_week)
   # X_train['year'] = X_train.Date.apply(create_year)
    X_test['month'] = X_test.Date.apply(create_month)
    X_test['day'] = X_test.Date.apply(create_day)
    X_test['week'] = X_test.Date.apply(create_week)
  #  X_test['year'] = X_train.Date.apply(create_year)

    print(X_train.head(5))

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


    # Merge with weather data
    X_train = X_train.merge(weather, on='Date')
    X_test = X_test.merge(weather, on='Date')
    X_train = X_train.drop(['Date'], axis = 1)
    X_test = X_test.drop(['Date'], axis = 1)

    return X_train, y_train, X_test, sample

def preProcessData_MergeClosest():
    # Load dataset
    X_train = pd.read_csv('input/train.csv')
    X_test = pd.read_csv('input/test.csv')
    sample = pd.read_csv('input/sampleSubmission.csv')
    weather = pd.read_csv('input/weather.csv')
    spray = pd.read_csv('input/spray.csv')

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

    #def create_year(x):
     #   return x.split('-')[0]

    X_train['month'] = X_train.Date.apply(create_month)
    X_train['day'] = X_train.Date.apply(create_day)
    X_train['week'] = X_train.Date.apply(create_week)
    #X_train['year'] = X_train.Date.apply(create_year)
    X_test['month'] = X_test.Date.apply(create_month)
    X_test['day'] = X_test.Date.apply(create_day)
    X_test['week'] = X_test.Date.apply(create_week)
   # X_test['year'] = X_train.Date.apply(create_year)

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


    def merge_closest(DF1, DF2, DF3, on_col, x_col, y_col):
        #matrix1 = DF1.values
        #matrix2 = DF2.values
        tempDF = DF2.copy()
        i=0
        startTime = datetime.datetime.now()
        print("Merging...")
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
            #if i<5:
            #    print "DF2: " + str(df2_dist) + " DF3: " +str(df3_dist)
            #    i+=1
            #print type(df2_row)
            if df2_dist<=df3_dist:
                tempDF[tempDF[on_col]==df1_date] = np.array(df2_row)
            else:
                tempDF[tempDF[on_col]==df1_date] = np.array(df3_row)
            i+=1
        print("Done! Elapsed Time: " + str(datetime.datetime.now()-startTime))
        return tempDF


    # Merge with weather data
    mergedWeather = merge_closest(X_train, weather_stn1, weather_stn2,on_col='Date', x_col='Latitude', y_col='Longitude')
    X_train = X_train.merge(mergedWeather, on='Date')
    print(X_train.head(5))
    mergedWeather = merge_closest(X_test, weather_stn1, weather_stn2,on_col='Date', x_col='Latitude', y_col='Longitude')
    X_test = X_test.merge(mergedWeather, on='Date')
    print(X_train.head(5))
    X_train = X_train.drop(['Date'], axis = 1)
    X_test = X_test.drop(['Date'], axis = 1)

    return X_train, y_train, X_test, sample

    
X_train, y_train, X_test, sample = preProcessData()    

def logistic1(X_train, y_train): 
    lr = LogisticRegression(penalty='l2', C = 1000)
    #parameters = {'C':[.001, .01, 1,10,100,1000]}
    #clf = GridSearchCV(lr, parameters, scoring=cm_loss, cv=5)
    lr.fit(X_train, y_train)
    print('done')
    return lr

def logistic2(X_train, y_train):
    #parameters = {'C':[.001, .01, 1]}
    lr2 = LogisticRegression(penalty = 'l1', C = 100000)
    #dlf = GridSearchCV(lr2, parameters, scoring =cm_loss, cv=5)
    lr2.fit(X_train, y_train)
    return lr2

def Ada(X_train, y_train):
    #parameters = {'n_estimators':[50, 500, 5000], 'learning_rate': [.001, .01, .1]}
    ada = AdaBoostClassifier(n_estimators =  5000, learning_rate = 1)
    #alf = GridSearchCV(ada, parameters, scoring = cm_loss, cv =3)
    ada.fit(X_train, y_train)
    return ada
    
def cm_loss(estimator, X, y):
    predictions = estimator.predict(X)
    cm = confusion_matrix(y, predictions)
    print(cm)
    totals = np.sum(cm, 1)
    no_acc = cm[0, 0] / totals[0]
    yes_acc = cm[1, 1] / totals[1]
    return (no_acc + yes_acc)/2
    
def dec(X_train, y_train):
    clf = tree.DecisionTreeClassifier(min_samples_split=5000, random_state=30)
    clf.fit(X_train, y_train)
    return clf

#rf = RandomForestClassifier(n_jobs=-1, min_samples_split = 1)
#parameters = {'n_estimators':[500, 1000, 2000]}
#clf = grid_search.GridSearchCV(rf, parameters, cv=5)
#clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1)
#clf.fit(X_train, y_train)
#6.68

#clf2 = ExtraTreesClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=1)
#clf2.fit(X_train, y_train)
#6.73
#from sklearn.linear_model import LogisticRegression
#rand = RandomTreesEmbedding(n_jobs=-1, n_estimators=1000, min_samples_split=1)

clf1 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1250, max_features = 2, loss = 'deviance', max_depth = 1, subsample = 1, )
clf1.fit(X_train, y_train)

clf2 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1250, max_features = 4, loss = 'deviance', max_depth = 3, subsample = 1, )
clf2.fit(X_train, y_train)

clf3 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1250, max_features = 4, loss = 'deviance', max_depth = 2, subsample = 1, )
clf3.fit(X_train, y_train)

clf4 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1250, max_features = 8, loss = 'deviance', max_depth = 7, subsample = 1, )
clf4.fit(X_train, y_train)

clf5 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1250, max_features = 16, loss = 'deviance', max_depth = 15, subsample = 1, )
clf5.fit(X_train, y_train)

clf6 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1250, max_features = 30, loss = 'deviance', max_depth = 29, subsample = 1, )
clf6.fit(X_train, y_train)
#.606

#clf3 = GradientBoostingClassifier(learning_rate = .075, n_estimators=1000, max_features = 4, loss = 'deviance', max_depth = 3, subsample = 1, )
#clf3.fit(X_train, y_train)
#.685

#clf4 = AdaBoostClassifier(learning_rate = .075, n_estimators=1000)
#clf4.fit(X_train, y_train)
#.685

#clf4 = svm.SVC(kernel='poly', probability=True, C=1.0)
#clf4.fit(X_train, y_train)
#Broken

#from sklearn.linear_model import LogisticRegression

#rand = RandomTreesEmbedding(n_jobs=-1, n_estimators=1000, min_samples_split=1)
#clf3 = rt_lm = LogisticRegression()
#pipeline = make_pipeline(rand, clf3)
#pipeline.fit(X_train, y_train)

#clf5 = svm.SVC(kernel='rbf', probability=True, C=1.0)
#clf5.fit(X_train, y_train)
#.560

predictions1 = clf1.predict_proba(X_test)[:,1]
sample['WnvPresent'] = predictions1
sample.to_csv('Visualization1.csv', index=False)

predictions2 = clf2.predict_proba(X_test)[:,1]
sample['WnvPresent'] = predictions2
sample.to_csv('Visualization2.csv', index=False)

predictions3 = clf3.predict_proba(X_test)[:,1]
sample['WnvPresent'] = predictions3
sample.to_csv('Visualization3.csv', index=False)

predictions4 = clf4.predict_proba(X_test)[:,1]
sample['WnvPresent'] = predictions4
sample.to_csv('Visualization4.csv', index=False)

predictions5 = clf5.predict_proba(X_test)[:,1]
sample['WnvPresent'] = predictions5
sample.to_csv('Visualization5.csv', index=False)

predictions6 = clf6.predict_proba(X_test)[:,1]
sample['WnvPresent'] = predictions6
sample.to_csv('Visualization6.csv', index=False)

#glf = DecisionTreeClassifier()
#def cm_loss(estimator, X, y):
#    predictions = estimator.predict(X)
#    cm = confusion_matrix(y, predictions)
#    totals = np.sum(cm, 1)
#    no_acc = cm[0, 0] / totals[0]
#    yes_acc = cm[1, 1] / totals[1]
#    return (no_acc + yes_acc)/2


#lr = LogisticRegression(penalty='l2')
#parameters = {'C':[.001, .01, 1]}
#clf = GridSearchCV(lr, parameters, scoring=cm_loss, cv=5)
#clf.fit(X_train, labels)
#lr2 = LogisticRegression(penalty = 'l1')
#dlf = GridSearchCV(lr2, parameters, scoring =cm_loss, cv=5)
#dlf.fit(X_train, labels)

# create predictions and submission file
#predictions = clf.predict_proba(X_test)[:,1]
#sample['WnvPresent'] = predictions
#sample.to_csv('Outcome.csv', index=False)
#predicts = dlf.predict_proba(X_test)[:,1]
#sample['WnvPresent'] = predicts
#sample.to_csv('Outcome2.csv', index=False)
#predicts = glf.predict_proba(X_test)[:,1]
#sample['WnvPresent'] = predicts
#sample.to_csv('Outcome3.csv', index=False)