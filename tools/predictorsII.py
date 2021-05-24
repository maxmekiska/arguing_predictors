import numpy as np
import pandas as pd

from neuralprophet import NeuralProphet



class UnivariatePredictorII:

    def __init__(self, data, future: int) -> object:
        self.data = self.__data_prep(data)
        self.future = future

    def __data_prep(self, data):
        data = data.rename(columns={data.columns[0]: 'y'}, inplace = False)
        data.index.names = ['df']
        return data



