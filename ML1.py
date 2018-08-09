import numpy as np
import pandas as pd
import math, quandl
from sklearn import preprocessing , cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle as pc

df = quandl.get("WIKI/GOOGL")
df = df[["Adj. Open","Adj. High","Adj. Low", "Adj. Close","Adj. Volume"]]
df.fillna(0,inplace=True)

forecast_label = "Adj. Close"
forecast_len = int(math.ceil(0.01*len(df)))

Y_val = df[forecast_label].shift(-forecast_len)
Y_val.dropna(inplace=True)
X_val = df.drop("Adj. Close",1)
X_val = X_val.iloc[:-forecast_len]
Y_val = np.array(Y_val)
X_val = np.array(X_val)

#print(len(X_val),len(Y_val))
#print(Y_val.tail())
X_val = preprocessing.scale(X_val)

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X_val,Y_val,test_size=0.3)

linear = LinearRegression()
linear.fit(X_train,Y_train)

with open("linearregression.pickle","wb") as f:
    pc.dump(linear,f)

pickle_load = open("linearregression.pickle","rb")
linear = pc.load(pickle_load)

confidence = linear.score(X_test,Y_test)
print(confidence)
