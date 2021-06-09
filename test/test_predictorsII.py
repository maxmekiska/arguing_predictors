import pandas as pd
import numpy as np
import unittest

from pandas.testing import assert_frame_equal

def data_prep(data):
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

d = {'Date': ['2010-02-01','2010-02-02', '2010-02-03', '2010-02-04', '2010-02-05', '2010-02-08', '2010-02-09', '2010-02-10'],'Close': [6.954642773, 6.994999886, 7.115356922, 6.858929157, 6.980713844, 6.932857037, 7.00678587, 6.968571186]}
data = pd.DataFrame(data=d)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

d2 = {'ds': ['2010-02-01','2010-02-02', '2010-02-03', '2010-02-04', '2010-02-05', '2010-02-08', '2010-02-09', '2010-02-10'],'y': [6.954642773, 6.994999886, 7.115356922, 6.858929157, 6.980713844, 6.932857037, 7.00678587, 6.968571186]}
data_solution = pd.DataFrame(data=d2)
data_solution['ds'] = pd.to_datetime(data_solution['ds'])

class Testing(unittest.TestCase):

    def test_data_prep(self):
        assert_frame_equal(data_prep(data), data_solution)

if __name__ == '__main__':
    unittest.main()
