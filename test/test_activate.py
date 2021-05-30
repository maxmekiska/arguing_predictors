import sys
sys.path.append('../')

import pandas as pd
from pandas.testing import assert_frame_equal
import unittest
from system.activate import *



d = {'Date': ['2010-02-01','2010-02-02', '2010-02-03', '2010-02-04', '2010-02-05', '2010-02-08', '2010-02-09', '2010-02-10'], 'High': [7, 7.011428833, 7.150000095, 7.084642887, 7, 7.067142963, 7.053571224, 7.021429062]}
data = pd.DataFrame(data=d)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

d1 = {'Date': ['2010-02-01','2010-02-02', '2010-02-03'], 'High': [7, 7.011428833, 7.150000095]}
solution = pd.DataFrame(data=d1)
solution['Date'] = pd.to_datetime(solution['Date'])
solution.set_index('Date', inplace=True)

d2 = {'Date': ['2010-02-04', '2010-02-05'], 'High': [7.084642887, 7]}
solution2 = pd.DataFrame(data=d2)
solution2['Date'] = pd.to_datetime(solution2['Date'])
solution2.set_index('Date', inplace=True)

d3 = {'High': [7, 7.011428833, 7.150000095]}
data2 = pd.DataFrame(data=d3)

class Testing(unittest.TestCase):

    def test_data_prep(self):
        assert_frame_equal(data_prep(data, 3, 2)[0], solution)

    def test_data_prep_1(self):
        assert_frame_equal(data_prep(data, 3, 2)[1], solution2)

    def test_set_same_index(self):
        assert_frame_equal(set_same_index(data2, solution), solution)
    



if __name__ == '__main__':
    unittest.main()

