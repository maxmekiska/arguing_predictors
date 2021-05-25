import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuralprophet import NeuralProphet



class UnivariatePredictorII:
    '''Implements the Facebooks Neural Prophet and Prophet libraries to forecast time series. This class is mainly a wrapper to ease usability.

        Methods
        -------
        __data_prep(self, data):
            Private method to shape input data for predictor ingestion.
        fit_model(self, epochs: int, frequency: str):
            Fits the model and validates during fitting process.
        show_performance(self):
            Plots the model performance.
        predict(self):
            Outputs prediction values.
    '''

    def __init__(self, data, future: int) -> object:
        '''
            Parameters:
                data (df): Input data onto which future predictions will be made. Next date after the last date in data is the first prediction value of model.
                future (int): How many days will be predicted into the future.
        '''
        self.data = self.__data_prep(data)
        self.future = future

    def __data_prep(self, data):
        '''Private function to prepare intake data into format that is digestible by the Neural Prophet library. The final format is a DataFrame with column 1 name = 'ds' containing date values and column 2 name = 'y' containing the data points.
            Parameters:
                data (df): Original non-formatted time series DataFrame

            Returns:
                data (df): Modified formatted time series DataFrame
        '''
        data = data.rename(columns={data.columns[0]: 'y'}, inplace = False)
        data = data.reset_index()
        data = data.rename(columns={'Date': 'ds'}, inplace = False)
        return data

    def fit_model(self, epochs: int, frequency: str):
        '''Method that implements the training and validation process of the model.

            Parameters:
                epochs (int): Number of epochs to train the model.
                frequency (str): Time series data frequency. For example: Daily = 'D'
        '''
        self.model = NeuralProphet()
        self.details = self.model.fit(self.data, epochs = epochs, validate_each_epoch=True, freq = frequency)

    def show_performance(self):
        '''Plots two graphs.
        1. Models mean average error of trainings and validation data.
        2. Models smooth L1 loss of trainings and validation data.
        '''
        information = self.details

        plt.subplot(1, 2, 1)
        plt.plot(information['MAE'])
        plt.plot(information['MAE_val'])
        plt.title('Model Mean Average Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')


        plt.subplot(1, 2, 2)
        plt.plot(information['SmoothL1Loss'])
        plt.plot(information['SmoothL1Loss_val'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.tight_layout()
        plt.show()

    def predict(self):
        '''Returns the forecasted values starting from the next data point from the very last data point in data provided.

            Returns:
                (df): Forecast for data provided.
        '''
        prediction = self.model.make_future_dataframe(self.data, periods=self.future)
        output = self.model.predict(prediction)
        return output['yhat1']


