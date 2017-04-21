import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import logging
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys

# fix random seed for reproducibility
np.random.seed(7)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# load the y
logger.debug('Reading in csv')
df = pd.read_csv('fb_stocks_small.csv')
look_back = 1
# Shift prices to predict the following day
df['close_tomorrow'] = df['close'].shift(-look_back)
df = df.drop(df.tail(look_back).index)
# Split test and train
y = df['close_tomorrow']
# X = df.drop(['close_tomorrow', 'ticker', 'date'], 1)
X = df[['open', 'close']]

# normalize the y
logger.debug('Formatting data')
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(y)
X['close'] = scaler.fit_transform(X['close'])
X['open'] = scaler.fit_transform(X['open'])

# split into train and test sets
train_size = int(0.33 * len(X))
test_size = len(y) - train_size
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[-test_size:]
y_test = y[-test_size:]

# reshape input to be [samples, time steps, features]
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# create and fit the LSTM network
logger.debug('Creating LSTM')
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, len(X.columns))))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
# make predictions
logger.debug('Predicting')
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(y)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = np.concatenate(trainPredict)
# shift test predictions for plotting
testPredictPlot = np.empty_like(y)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+look_back-1:len(y)] = np.concatenate(testPredict)

# plot baseline and predictions
plt.plot(scaler.inverse_transform(y))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print arg

if __name__ == "__main__":
    main()