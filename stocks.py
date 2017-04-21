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
scaler = MinMaxScaler(feature_range=(0, 1))

window = 5
epochs = 20
batch_size = 1
lstm_neurons = 10


def load_data():
    cols = ['open',
            'high',
            'low',
            'close',
            'volume',
            'ex-dividend',
            'split_ratio',
            'adj_open',
            'adj_high',
            'adj_low',
            'adj_close',
            'adj_volume']
    df = pd.read_csv('fb_stocks.csv')
    y = df['close'].shift(-1)[:-1]
    X = df[cols]
    # Normalize
    y = scaler.fit_transform(y)
    for c in cols:
        X[c] = scaler.fit_transform(X[c])
    return X, y


def format_data(X, y):
    # Create [samples, time steps, features]
    samples = []
    for i in range(window, len(y)):
        samples.append([X.loc[i-t] for t in reversed(range(window))])
    X = samples
    # split into train and test sets
    train_size = int(0.90 * len(X))
    test_size = len(y) - train_size
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    # reshape input to be [samples, time steps, features]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    return X_train, y_train, X_test, y_test


def train(X_train, y_train):
    model = Sequential()
    model.add(LSTM(
        lstm_neurons,
        batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2]),
        stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(epochs):
        print 'batch', i+1
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            verbose=2,
            shuffle=False,
            validation_split=0.33)
        model.reset_states()
    return model


def predict(model, X_train, X_test):
    trainPredict = model.predict(X_train, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(X_test, batch_size=batch_size)
    return trainPredict, testPredict


def graph(trainPredict, testPredict, y, y_train, y_test):
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])
    testPredict = scaler.inverse_transform(testPredict)
    y_test = scaler.inverse_transform([y_test])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(y)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[1:len(trainPredict)+1] = np.concatenate(trainPredict)
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(y)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict):len(y)] = np.concatenate(testPredict)

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
    graph(train_p, test_p, y, y_train, y_test)

if __name__ == "__main__":
    main()
