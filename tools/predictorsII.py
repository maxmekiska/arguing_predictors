import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuralprophet import NeuralProphet



class UnivariatePredictorII:

    def __init__(self, data, future: int) -> object:
        self.data = self.__data_prep(data)
        self.future = future

    def __data_prep(self, data):
        data = data.rename(columns={data.columns[0]: 'y'}, inplace = False)
        data = data.reset_index()
        data = data.rename(columns={'Date': 'ds'}, inplace = False)
        return data

    def fit_model(self, epochs: int, frequency: str):
        self.model = NeuralProphet()
        self.details = self.model.fit(self.data, epochs = epochs, validate_each_epoch=True, freq = frequency)

    def show_performance(self):
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
        prediction = self.model.make_future_dataframe(self.data, periods=self.future)
        output = self.model.predict(prediction)
        return output['yhat1']


