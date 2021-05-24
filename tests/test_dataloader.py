import sys
sys.path.append('../')

import pandas as pd
from pandas.util.testing import assert_frame_equal
import unittest
from tools.dataloader import *




d = {'Date': ['2010-02-01','2010-02-02', '2010-02-03', '2010-02-04', '2010-02-05', '2010-02-08', '2010-02-09', '2010-02-10'], 'High': [7, 7.011428833, 7.150000095, 7.084642887, 7, 7.067142963, 7.053571224, 7.021429062], 'Low': [6.83214283, 6.906428814, 6.943571091, 6.841785908, 6.816071033, 6.928571224, 6.955357075, 6.937857151], 'Open': [6.870357037, 6.996786118, 6.970356941, 7.026071072, 6.879642963, 6.988928795, 7.014999866, 6.996070862], 'Close': [6.954642773, 6.994999886, 7.115356922, 6.858929157, 6.980713844, 6.932857037, 7.00678587, 6.968571186], 'Volume': [749876400, 698342400, 615328000, 757652000, 850306800, 478270800, 632886800, 370361600], 'Adj Close': [5.980317116, 6.015021324, 6.118516922, 5.898012161, 6.002737999, 5.96158123, 6.025155067, 5.992293358]}
data = pd.DataFrame(data=d)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data_high = pd.DataFrame(data['High'])
data_low = pd.DataFrame(data['Low'])
data_open = pd.DataFrame(data['Open'])
data_close = pd.DataFrame(data['Close'])
data_volume = pd.DataFrame(data['Volume'])
data_adj = pd.DataFrame(data['Adj Close'])
data_stat = data.describe()


d_alt = {'Date': ['2010-02-01','2010-02-02', '2010-02-03', '2010-02-04', '2010-02-05', '2010-02-08', '2010-02-09', '2010-02-10'], 'Open': [6.87, 7.00, 6.97, 7.03, 6.88, 6.99, 7.01, 7.00], 'High': [7.00, 7.01, 7.15, 7.08, 7.00, 7.07, 7.05, 7.02], 'Low': [6.83, 6.91, 6.94, 6.84, 6.82, 6.93, 6.96, 6.94], 'Close': [6.95, 6.99, 7.12, 6.86, 6.98, 6.93, 7.01, 6.97], 'Volume': [749874432, 698341952, 615327680, 757651776, 850306240, 478268960, 632884736, 370361248], 'Currency': ['USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD', 'USD']}
data_alt = pd.DataFrame(data=d_alt)
data_alt['Date'] = pd.to_datetime(data_alt['Date'])
data_alt.set_index('Date', inplace=True)



class Testing(unittest.TestCase):

    def test_get_data(self):
        assert_frame_equal(output, data)
    def test_get_high(self):
        assert_frame_equal(high, data_high)
    def test_get_low(self):
        assert_frame_equal(low, data_low)
    def test_get_open(self):
        assert_frame_equal(open_, data_open)
    def test_get_close(self):
        assert_frame_equal(close_, data_close)
    def test_get_volume(self):
        assert_frame_equal(volume, data_volume)
    def test_get_adj(self):
        assert_frame_equal(adj, data_adj)
    def test_statistics(self):
        assert_frame_equal(stat, data_stat)

    def test_alternative_get_data(self):
        assert_frame_equal(output_alternative, data_alt)

if __name__ == '__main__':
    loaded = DataLoader('aapl', '2010-02-01', '2010-02-10')
    output = loaded.get_data()
    high = loaded.get_high()
    low = loaded.get_low()
    open_ = loaded.get_open()
    close_ = loaded.get_close()
    volume = loaded.get_volume()
    adj = loaded.get_adjclose()
    stat = loaded.statistics()

    alternative = AlternativeDataLoader('AAPL', 'United States', '01/02/2010', '10/02/2010', 'stock')
    output_alternative = alternative.get_data()
    unittest.main()



