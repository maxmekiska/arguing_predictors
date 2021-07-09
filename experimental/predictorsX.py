''' References:
Code insperations have been taken from:
Brwonlee, J., 2016. Display deep learning model training history in keras [Online]. Available from:
https://machinelearningmastery.com/display-deep-
learning-model-training-history-in-keras/.
Brwonlee, J., 2018a. How to develop convolutional neural network models for time series forecasting [Online]. Available from: 
https://machinelearningmastery.com/how-to-develop-convolutional-
neural-network-models-for-time-series-forecasting/.
Brwonlee, J., 2018b. How to develop lstm models for time series forecasting [Online]. Available from: 
https://machinelearningmastery.com/how-to-develop-
lstm-models-for-time-series-forecasting/.
Brwonlee, J., 2018c. How to develop multilayer perceptron models for time series forecasting [Online]. Available from: 
https://machinelearningmastery.com/how-to-develop-multilayer-
perceptron-models-for-time-series-forecasting/.
'''

import matplotlib.pyplot as plt
from numpy import array
from numpy import reshape
from numpy import empty
from numpy import dstack

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
from keras.layers import Dropout

from keras.layers import Bidirectional

class BasicMultivariatePredictor:
    '''Implements neural network based univariate multipstep predictors.
        
        Methods
        -------
        _data_prep(self, stockdata: DataFrame) -> array:
            Private method to extract features and convert DataFrame to an array.
        _sequence_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> array:
            Private method to prepare data for predictor ingestion.
        _multistep_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> array:
            Private method to apply sequence_prep to every feature.
        set_model_id(self, name: str)
            Setter method to change model id name.
        create_lstm(self):
            Builds LSTM structure.
        create_cnn(self):
            Builds CNN structure.
        create_bilstm(self):
            Builds bidirectional LSTM structure.
        fit_model(self, epochs: int, show_progress: int = 1):
            Training the in the prior defined model. Count of epochs need to be defined.
        model_blueprint(self):
            Print blueprint of layer structure.
        show_performance(self):
            Evaluate and plot model performance.
        predict(self, data: array):
            Takes in input data and outputs model forecasts.
        save_model(self):
            Saves current Keras model to current directory.
        load_model(self, location: str):
            Load model from location specified.
    '''
    def __init__(self, steps_past: int, steps_future: int, data = pd.DataFrame()) -> object:
        '''
            Parameters:
                steps_past (int): Steps predictor will look backward.
                steps_future (int): Steps predictor will look forward.
                data (DataFrame): Input data for model training.
        '''
        if len(data) > 0:
            self.data = self._data_prep(data)
            self.input_x, self.input_y = self._multistep_prep(self.data, steps_past, steps_future)
        else:
            self.data = data
            
        self.model_id = '' # to identify model (example: name)

    def _data_prep(self, stockdata: DataFrame) -> array:
        ''' Private method to extract features and convert DataFrame to an array. Extracts: Adj Close, Open, High and Low features. 

                Parameters:
                    stockdata (DataFrame): DataFrame containing multi-feature stock data.

                Retruns:
                    data (array): Array containing sequences of selected features.
                    
        '''
        adj = list(stockdata['Adj Close'])
        open_ = list(stockdata['Open'])
        high = list(stockdata['High'])
        low = list(stockdata['Low'])
        
        data = [adj, open_, high, low]
        data = array(data)
        
        return data        
        
    def _sequence_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> array:
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
        if length == 0:
            return (empty(shape=[steps_past, steps_past]), 0)
        X = []
        y = []
        if length <= steps_past:
            raise ValueError('Input sequence is equal to or shorter than steps to look backwards')
        if steps_future <= 0:
            raise ValueError('Steps in the future need to be bigger than 0')
    
        for i in range(length):
            last = i + steps_past
            if last > length - steps_future:
                X.append(input_sequence[i:last])
                y.append(input_sequence[last-1:last-1 + steps_future])
                break
            X.append(input_sequence[i:last])
            y.append(input_sequence[last-1:last-1 + steps_future]) # modification to use correct target y sequence
        y = array(y)
        X = array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y
    
    def _multistep_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> array:
        '''This function prepares input sequences into a suitable input format for a multivariate multistep model. The first seqeunce in the array needs to be the target variable y.

                Parameters:
                    input_sequence (array): Sequence that contains time series in array format
                    steps_past (int): Steps the predictor will look backward
                    steps_future (int): Steps the predictor will look forward
                Returns:
                    X (array): Array containing all looking back sequences
                    y (array): Array containing all looking forward sequences   
        '''
        X = []
        Y = []
        for i in range(len(input_sequence)):
            if i ==0: # target variable should be first sequence
                _, y = self._sequence_prep(input_sequence[0], steps_past, steps_future)
                Y.append(y)
                continue # skip since target column not requiered in X array
            x, _ = self._sequence_prep(input_sequence[i], steps_past, steps_future)
            X.append(x)
        X = dstack(X)
        Y = Y[0] # getting array out of list
        return X, Y
    
    def set_model_id(self, name: str):
        '''Setter method to change model id field.
        '''
        self.model_id = name

    def create_mlp(self):
        '''Creates MLP model by defining all layers with activation functions, optimizer, loss function and evaluation metrics. 
        '''
        self.set_model_id('MLP')

        self.dimension = (self.input_x.shape[1] * self.input_x.shape[2])

        self.input_x = self.input_x.reshape((self.input_x.shape[0], self.dimension)) # necessary to account for different shape input for MLP compared to the other models.

        self.model = Sequential()
        self.model.add(Dense(50, activation='relu', input_dim = self.dimension))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])        

    def create_lstm(self):
        '''Creates LSTM model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('LSTM')

        self.model = Sequential()
        self.model.add(LSTM(40, activation='relu', return_sequences=True, input_shape=(self.input_x.shape[1], self.input_x.shape[2])))
        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        
    def create_cnn(self):
        '''Creates the CNN model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN')

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.input_x.shape[1], self.input_x.shape[2])))
        self.model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def create_bilstm(self):
        '''Creates a bidirectional LSTM model by defining all layers with activation functions, optimizer, loss function and evaluation matrics.
        '''
        self.set_model_id('Bidirectional LSTM')

        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=(self.input_x.shape[1], self.input_x.shape[2])))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def fit_model(self, epochs: int, show_progress: int = 1):
        '''Trains the model on data provided. Performs validation. 
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
        dimension = (data.shape[0] * data.shape[1]) # MLP case

        try:
            data = data.reshape((1, data.shape[0], data.shape[1])) # All other models
            y_pred = self.model.predict(data, verbose=0)

        except:
            data = data.reshape((1, dimension)) # MLP case
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

def pred_input(stockdatapred: DataFrame, steps_past: int, target: str) -> array:
    '''Helper function to prepare data input to enable model to make future predictions.
    
        Parameters:
            stockdatapred (DataFrame): DataFrame containing all features that shall be considered for predicting the future including the target feature.
            steps_past (int): How much the model needs to look back to make a prediction.
            target (str): Target variable name.
            
        Returns:
            X (array): Array containing data the model uses to predict into the future.
    
    '''
    stockdatapred = stockdatapred.drop(target, axis=1)
    stockdatapred = stockdatapred.iloc[0:steps_past]
    X = []
    for i in range(len(stockdatapred)):
        X.append(list(stockdatapred.iloc[i]))
    return array(X)
