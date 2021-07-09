''' References:
Code insperations have been taken from:

Brwonlee, J., 2016. Display deep learning model training history in keras [Online]. Available from:
https://machinelearningmastery.com/display-deep-
learning-model-training-history-in-keras/.

Brwonlee, J., 2018b. How to develop lstm models for time series forecasting [Online]. Available from: 
https://machinelearningmastery.com/how-to-develop-
lstm-models-for-time-series-forecasting/.
'''
import matplotlib.pyplot as plt
from numpy import array
from numpy import reshape
import pandas as pd
from pandas import DataFrame
import os

from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

class HybridUnivariatePredictor:
    '''Implements neural network based univariate multipstep hybrid predictors.
        
        Methods
        -------
        _sequence_prep(input_sequence: array, steps_past: int, steps_future: int) -> [(array, array, int)]:
            Private method to prepare data for predictor ingestion.
        set_model_id(self, name: str):
            Setter method to change model id name.
        create_cnnlstm(self):
            Builds CNN-LSTM structure.
        fit_model(self, epochs: int, show_progress: int = 1):
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
    def __init__(self, sub_seq: int, steps_past: int, steps_future: int, data = pd.DataFrame()) -> object:
        '''
            Parameters:
                sub_seq (int): Further division of given steps a predictor will look backward.
                steps_past (int): Steps predictor will look backward.
                steps_future (int): Steps predictor will look forward.
                data (array): Input data for model training. Default is empty to enable loading modles.
        '''
        self.data = array(data) 
        self.sub_seq = sub_seq
        self.input_x, self.input_y, self.modified_back = self._sequence_prep(data, sub_seq, steps_past, steps_future)

        self.model_id = '' # identify model (example: name)

    def _sequence_prep(self, input_sequence: array, sub_seq: int, steps_past: int, steps_future: int) -> [(array, array, int)]:
        '''Prepares data input into X and y sequences. Lenght of the X sequence is dertermined by steps_past while the length of y is determined by steps_future. In detail, the predictor looks at sequence X and predicts sequence y.

            Parameters:
                input_sequence (array): Sequence that contains time series in array format
                sub_seq (int): Further division of given steps a predictor will look backward.
                steps_past (int): Steps the predictor will look backward
                steps_future (int): Steps the predictor will look forward

            Returns:
                X (array): Array containing all looking back sequences
                y (array): Array containing all looking forward sequences
                modified_back (int): Modified looking back sequence length
        '''
        length = len(input_sequence)
        if length == 0:
            return (0, 0, steps_past // sub_seq)
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
        modified_back = X.shape[1]//sub_seq
        X = X.reshape((X.shape[0], sub_seq, modified_back, 1))
        return X, y, modified_back # special treatment to account for sub sequence division

    def set_model_id(self, name: str):
        '''Setter method to change model id field.
        '''
        self.model_id = name

    def create_cnnlstm(self):
        '''Creates CNN-LSTM hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics. 
        '''
        self.set_model_id('CNN-LSTM')

        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(LSTM(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Experimental model
    def create_cnnbilstm(self):
        '''Creates CNN-Bidirectional-LSTM hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics. 
        '''
        self.set_model_id('CNN-Bi-LSTM')

        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True)))
        self.model.add(LSTM(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit_model(self, epochs: int, show_progress: int = 1):
        '''Trains the model on data provided. Perfroms validation. 
            Parameters:
                epochs (int): Number of epochs to train the model. 
                show_progress (int): Prints training progress.
        '''
        self.details = self.model.fit(self.input_x, self.input_y, validation_split=0.20, batch_size = 10, epochs = epochs, verbose=show_progress)
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
        plt.title(self.model_id + ' Model Mean Squared Error')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(information.history['loss'])
        plt.plot(information.history['val_loss'])
        plt.title(self.model_id + ' Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.tight_layout()
        plt.show()

    def predict(self, data: array) -> DataFrame:
        '''Takes in a sequence of values and outputs a forecast.

            Parameters:
                data (array): Input sequence which needs to be forecasted.

            Returns:
                (DataFrame): Forecast for sequence provided.
        '''
        data = array(data)

        data = data.reshape((1, self.sub_seq, self.modified_back, 1))
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
        self.model = keras.models.load_model(location)
