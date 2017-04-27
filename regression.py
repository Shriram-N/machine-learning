#supervised ML everything boils down to features and labels
# each column is a feature but i have to choose the right features to work with
#features are like attributes that make up labels,hopefully it will help in some kind of prediction into the future
#simplify data as much as possible
import pandas as pd
import quandl
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df=quandl.get('WIKI/GOOGL')

#print (df.head())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

#label seems to be the one which we are gonna predict or forecast
#features are the descriptive attributes

forecast_col='Adj. Close'

#in order to save loss of data and to handle missing feature data value we will fill the missing data with -99999 which is generally an outlier

df.fillna(-99999,inplace=True)

forecast_out =int(math.ceil(0.01*len(df)))

df['label'] = df['Adj. Close'].shift(forecast_out)

df.dropna(inplace =True)

print df.head()

#practice using X as features and Y as labels
X= np.array(df.drop(['label'],1))
Y=np.array(df['label'])

#we will do pre-processing of the data before testing and training
#we need our feature values to be in the range of 1 to -1 this will speed up our processing time and help with accuracy

X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
y = np.array(df['label'])


#next is training and testing .we will split the data into 75% for training and 25% for testing .By testing we get the confidence and accuracy scores which tells us how much more data we need to train.here we use cross validation which will also shuffle the data for us

X_train,X_test,y_train,y_test =cross_validation.train_test_split(X,y,test_size=0.2)

#we will now use the classifier
#If it has n_jobs, you have an algorithm that can be threaded for high performance
#clf = svm.SVR()
'''
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
'''
clf = LinearRegression(n_jobs=-1)

# we will now train the data with .fit method
clf.fit(X_train, y_train)

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
confidence=clf.score(X_test,y_test)

pickle_in =open('linearregression.pickle', 'rb')
clf =pickle.load(pickle_in)

forecast_set = clf.predict(X_lately)

print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


