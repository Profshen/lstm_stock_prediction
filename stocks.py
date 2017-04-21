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


np.random.seed(7)
look_back = 1
scaler = MinMaxScaler(feature_range=(0, 1))


def load_data():
    df = pd.read_csv('fb_stocks_small.csv')
    df['close_tomorrow'] = df['close']
    y = df['close_tomorrow']
    X = df[['open', 'close']]
    # Normalize
    y = scaler.fit_transform(y)
    X['close'] = scaler.fit_transform(X['close'])
    X['open'] = scaler.fit_transform(X['open'])
    return X, y


def format_data(X, y):
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
    # Need to create sample for each window of length lookback
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    return X_train, y_train, X_test, y_test


def train(X_train, y_train):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)
    return model


def predict(model, X_train, X_test):
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    return trainPredict, testPredict


def graph(trainPredict, testPredict, y):
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    # y_train = scaler.inverse_transform([y_train])
    testPredict = scaler.inverse_transform(testPredict)
    # y_test = scaler.inverse_transform([y_test])
    # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:, 0]))
    # print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))

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
    X, y = load_data()
    X_train, y_train, X_test, y_test = format_data(X, y)
    model = train(X_train, y_train)
    train_p, test_p = predict(model, X_train, X_test)
    graph(train_p, test_p, y)

if __name__ == "__main__":
    main()
