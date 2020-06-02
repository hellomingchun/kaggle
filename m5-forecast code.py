# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:55:38 2020

@author: Ming-Chun Chen
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

dataPath  = 'C:/Users/myPC/PycharmProjects/m5Forecasting/venv/kaggle/input/'
dt = pd.read_csv(dataPath + "/sales_train_validation.csv")

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

dt = downcast_dtypes(dt)
dt = dt.T
dt = dt[6:]

calendar = pd.read_csv(dataPath + "/calendar.csv")
eventDay = pd.DataFrame(np.zeros((1969,1)))

# "1" is assigned to the days before the event_name_1. Since "event_name_2" is rare, it was not added.
for x,y in calendar.iterrows():
    if((pd.isnull(calendar["event_name_1"][x])) == False):
           eventDay[0][x] = 1

#"eventDayTest" will be used as input for predicting (We will forecast the days 1913-1941)
eventDayTest = eventDay[1913:1941]
#"eventDay" will be used for training as a feature.
eventDay = eventDay[:1913]

eventDay.columns = ["eventDay"]
eventDay.index = dt.index

snapByStates = calendar[["snap_CA","snap_TX","snap_WI"]][:1913]
snapByStates.index = dt.index

n_lag = 7
dt = pd.concat([dt, eventDay, snapByStates], axis = 1)
dt.columns

sc = MinMaxScaler(feature_range = (0, 1))
train, test = dt[:1900], dt[-8:]
train = sc.fit_transform(train)
test = sc.transform(test)
# generator = TimeseriesGenerator(train, train, n_lag, batch_size=42)
train_data_gen = TimeseriesGenerator(train, train, length=n_lag, sampling_rate=1, stride=1, batch_size = 60)
valid_data_gen = TimeseriesGenerator(train, train, length=n_lag, sampling_rate=1, stride=1, batch_size = 18)
test_data_gen = TimeseriesGenerator(test, test, length=n_lag, sampling_rate=1, stride=1, batch_size = 1)
# print(len(generator))
# number of samples

# print each sample
for i in range(len(train_data_gen)):
	x, y = train_data_gen[i]

# Initialising the RNN
clear_session()
model = Sequential()
model.add(LSTM(units = 100, return_sequences = True, input_shape = (x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 400, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 400))
model.add(Dropout(0.2))
model.add(Dense(units = 30494, activation='relu'))

earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
adamOpti = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# Compiling the RNN
model.compile(optimizer = adamOpti, loss = 'mean_squared_error')
history = model.fit(train_data_gen, validation_data=valid_data_gen, steps_per_epoch=14, epochs=500, callbacks=[earlystopper], verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,0.05)
plt.show()

error = model.evaluate(test_data_gen)
y_test_pred = model.predict(test_data_gen)
y_test_pred = sc.inverse_transform(y_test_pred)[:,0:30490]
pred3 = y_test_pred.T

actual = dt[-1:]
actual2 = actual.T