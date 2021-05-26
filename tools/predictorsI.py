'''
References
----------
Code insperations have been taken from:
- https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
- https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/#:~:text=Visualize%20Model%20Training%20History%20in,from%20the%20collected%20history%20data.&text=A%20plot%20of%20loss%20on,validation%20datasets%20over%20training%20epochs.
- https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
- https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/ 
'''


import matplotlib.pyplot as plt
from numpy import array
from numpy import reshape
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


from keras.layers import Bidirectional


class BasicUnivariatePredictor:
    '''Implements neural network based univariate multipstep predictors.
        
        Methods
        -------
        sequence_prep(input_sequence: array, steps_past: int, steps_future: int) -> array:
            Private method to prepare data for predictor ingestion.
        create_lstm(self):
            Builds LSTM structure.
        create_cnn(self):
            Builds CNN structure.
        create_bilstm(self):
            Builds bidirectional LSTM structure.
        fit_model(self, epochs: int):
            Training the in the prior defined model. Count of epochs need to be defined.
        model_blueprint(self):
            Print blueprint of layer structure.
        show_performance(self):
            Evaluate and plot model performance.
        predict(self, data: array):
            Takes in input data and outputs model forecasts.
        save_model(self):
            Saves current ceras model to current directory.
        load_model(self, location: str):
            Load model from location specified.
    '''

    def __init__(self, data, steps_past: int, steps_future: int) -> object:
        '''
            Parameters:
                data (array): Input data for model training.
                steps_past (int): Steps predictor will look backward.
                steps_future (int): Steps predictor will look forward.
        '''
        self.data = array(data)
        self.input_x, self.input_y = self.__sequence_prep(data, steps_past, steps_future)
        
        self.model_id = ''


    def __sequence_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> array:
        '''Prepares data input into X and y sequences. Lenght of the X sequence is dertermined by steps_past while the length of y is determined by steps_future. In detail, the predictor looks at sequence X and predicts sequence y.

            Parameters:
                input_sequence (array): Sequence that contains time series in array format
                steps_past (int): Steps the predictor will look backward
                steps_future (int): Steps the predictor will look forward

            Returns:
                X (array): Array containing all looking back sequences
                y (array): Array containing all looking forward sequences
        '''
        length = len(input_sequence)
        X = []
        y = []
        if length <= steps_past:
            raise ValueError('Input sequence is equal to or shorter than steps to look backwards')
        if steps_future <= 0:
            raise ValueError('Steps in the future need to be bigger than 0')

        for i in range(length):
            last = i + steps_past
            if last > length - steps_future:
                break
            X.append(input_sequence[i:last])
            y.append(input_sequence[last:last + steps_future])
        y = array(y)
        X = array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def create_mlp(self):
        '''Creates MLP model by defining all layers with activation functions, optimizer, loss function and evaluation metrics. 
        '''
        self.model_id  = 'MLP'

        self.input_x = self.input_x.reshape((self.input_x.shape[0], self.input_x.shape[1])) # necessary to account for different shape input for MLP compared to the other models.

        self.model = Sequential()
        self.model.add(Dense(50, activation='relu', input_dim = self.input_x.shape[1]))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def create_lstm(self):
        '''Creates LSTM model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.model_id = 'LSTM'

        self.model = Sequential()
        self.model.add(LSTM(40, activation='relu', return_sequences=True, input_shape=(self.input_x.shape[1], 1)))
        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def create_cnn(self):
        '''Creates the CNN model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.model_id = 'CNN'

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.input_x.shape[1], 1)))
        self.model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def create_bilstm(self):
        '''Creates a bidirectional LSTM model by defining all layers with activation functions, optimizer, loss function and evaluation matrics.
        '''
        self.model_id = 'Bidirectional LSTM'

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=(self.input_x.shape[1], 1)))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit_model(self, epochs: int):
        '''Trains the model on data provided. Perfroms validation. 
            Parameters:
                epochs (int): Number of epochs to train the model. 
        '''
        self.details = self.model.fit(self.input_x, self.input_y, validation_split=0.20, batch_size = 10, epochs = epochs, verbose=1)
        return self.details

    def model_blueprint(self):
        '''Prints a summary of the models layer structure.
        '''
        self.model.summary()
    
    def show_performance(self):
        '''Plots two graphs.
        1. Models mean squared error of trainings and validation data.
        2. Models loss of trainings and validation data.
        '''

        information = self.details


        plt.subplot(1, 2, 1)
        plt.plot(information.history['mean_squared_error'])
        plt.plot(information.history['val_mean_squared_error'])
        plt.title('Model Mean Squared Error')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')


        plt.subplot(1, 2, 2)
        plt.plot(information.history['loss'])
        plt.plot(information.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.tight_layout()
        plt.show()

    def predict(self, data: array) -> array:
        '''Takes in a sequence of values and outputs a forecast.

            Parameters:
                data (array): Input sequence which needs to be forecasted.

            Returns:
                (array): Forecast for sequence provided.
        '''
        data = array(data)
        try:
            data = data.reshape((1, self.input_x.shape[1], 1))
            y_pred = self.model.predict(data, verbose=0)
        except:
            data = data.reshape((1, self.input_x.shape[1]))
            y_pred = self.model.predict(data, verbose=0)


        y_pred = y_pred.reshape(y_pred.shape[1], y_pred.shape[0])
            
        return pd.DataFrame(y_pred, columns=[f'{self.model_id}'])


    def save_model(self):
        '''Save the current model to the current directory.
        '''
        self.model.save(os.path.abspath(os.getcwd()))

    def load_model(self, location: str):
        '''Load a keras model from the path specified.

            Parameters:
                location (str): Path of keras model location
        '''
        self.model = keras.models.load_model(path)

