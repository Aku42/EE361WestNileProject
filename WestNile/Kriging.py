from PreProcessing import *
import pyKriging

if __name__ == '__main__':
    X_train = pd.read_csv('predata/trainmerged.csv')
    X_test = pd.read_csv('predata/testmerged.csv')
    sample = pd.read_csv('input/sampleSubmission.csv')
    y_train =pd.read_csv('input/train.csv')['WnvPresent']
    X_train['Day'] = 7*X_train['Week']+X_train['Day']
    X_test['Day'] = 7*X_test['Week']+X_test['Day']
    krig_train = X_train.ix[:,['Latitude','Longitude','Day']]
    krig_test = X_test.ix[:,['Latitude','Longitude','Day']]
    log_test = X_test.drop(['Latitude','Longitude','Day'],axis=1)
    log_train = X_train.drop(['Latitude','Longitude','Day'],axis=1)
    gp = LogisticRegression(C=100)
    gp.fit(log_train, y_train)
    train_pred = gp.predict(X=log_train)
    temp_pred = gp.predict(X=log_test)
    krig_train['logpred']=train_pred
    krig_test['logpred']=temp_pred

    sample['WnvPresent'] = temp_pred
    sample.to_csv('lograw.csv', index=False)
    print "Here1"

    gp = GaussianProcess(theta0=1e10)
    #gp = pyKriging.kriging(np.array(krig_train), np.array(y_train))
    print "Here"
    gp.fit(krig_train,y_train)
    krig1 = krig_test.loc[:len(krig_test)/4,:]
    predictions = gp.predict(krig1)
    print "Here"
    krig1 = krig_test.loc[(len(krig_test)/4)+1:2*(len(krig_test)/4),:]
    predictions = np.append(predictions,gp.predict(krig1))
    print "Here"
    krig1 = krig_test.loc[2*(len(krig_test)/4)+1:3*(len(krig_test)/4),:]
    predictions = np.append(predictions,gp.predict(krig1))
    print "Here"
    krig1 = krig_test.loc[3*(len(krig_test)/4)+1:,:]
    predictions = np.append(predictions,gp.predict(krig1))
    print "Here"

    sample['WnvPresent'] = predictions
    sample.to_csv('krigingraw.csv', index=False)

    sample = pd.read_csv('krigingraw.csv')
    predictions = np.array(sample['WnvPresent'])
    predictions *= (1/(predictions.max()-predictions.min()))
    predictions += .5
    predictions /= 2
    sample['WnvPresent'] = predictions
    sample.to_csv('kriging.csv', index=False)