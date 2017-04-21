from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class Simple_LSTM:

    def __init__(self, input_shape, window=1, num_neurons=5, batch_size=1):
        self.model = Sequential()
        self.model.add(LSTM(
            num_neurons,
            batch_input_shape=(batch_size, input_shape[0], input_shape[1]),
            stateful=True))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.batch_size = batch_size
        self.window = window
        self.num_neurons = num_neurons

    def train(self, X_train, y_train, epochs=1):
        for i in range(epochs):
            print 'batch', i+1
            self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=self.batch_size,
                verbose=2,
                shuffle=False,
                validation_split=0.33)
            self.model.reset_states()

    def predict(self, X_train, X_test, y_train, y_test, scaler):
        trainPredict = self.model.predict(X_train, batch_size=self.batch_size)
        self.model.reset_states()
        testPredict = self.model.predict(X_test, batch_size=self.batch_size)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        y_train = scaler.inverse_transform([y_train])
        testPredict = scaler.inverse_transform(testPredict)
        y_test = scaler.inverse_transform([y_test])
        # calculate root mean squared error
        trainScore = math.sqrt(
            mean_squared_error(y_train[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(
            mean_squared_error(y_test[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))
        return trainPredict, testPredict
