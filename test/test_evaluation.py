import pandas as pd
from pandas.testing import assert_frame_equal
import unittest
from tools.evaluation import *

d = {'Date': ['2010-02-01','2010-02-02', '2010-02-03', '2010-02-04', '2010-02-05'], 'Individual Predictor I': [10, 23, 12, 12, 23], 'Individual Predictor II': [10, 22, 13, 12, 22], 'Individual Predictor III': [9, 20, 12, 12, 20], 'Real Value': [9, 19, 12, 14, 21], 'Consensus Algorithm I': [12.3, 22.5, 13, 13, 21], 'Consensus Algorithm II': [9.2, 19.4, 12.1, 13, 22]}

data = pd.DataFrame(data=d)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

dmse = {'Algorithms': ['Individual Predictor I', 'Individual Predictor II', 'Individual Predictor III', 'Consensus Algorithm I', 'Consensus Algorithm II'], 'MSE': [5.00, 3.200, 1.200, 5.028, 0.442]}

solution_MSE = pd.DataFrame(data=dmse)

dmae = {'Algorithms': ['Individual Predictor I', 'Individual Predictor II', 'Individual Predictor III', 'Consensus Algorithm I', 'Consensus Algorithm II'], 'MAE': [1.80, 1.60, 0.80, 1.76, 0.54]}

solution_MAE = pd.DataFrame(data=dmae)

dmselog = {'Algorithms': ['Individual Predictor I', 'Individual Predictor II', 'Individual Predictor III', 'Consensus Algorithm I', 'Consensus Algorithm II'],
     'MSE Log': [0.014074802324481587, 0.011312646618774585, 0.005004488685004654, 0.023517290475069424, 0.001515796645209273]}

solution_MSELog = pd.DataFrame(data=dmselog)

d1 = {'Date': ['2010-02-01','2010-02-02', '2010-02-03'], 'High': [7, 7.011428833, 7.150000095]}
solution = pd.DataFrame(data=d1)
solution['Date'] = pd.to_datetime(solution['Date'])
solution.set_index('Date', inplace=True)

d3 = {'High': [7, 7.011428833, 7.150000095]}
data2 = pd.DataFrame(data=d3)


class Testing(unittest.TestCase):

    def test_set_same_index(self):
        assert_frame_equal(set_same_index(data2, solution), solution)

    def test_mse(self):
        assert_frame_equal(mse_score(data), solution_MSE)

    def test_mse_log(self):
        assert_frame_equal(mse_log_score(data), solution_MSELog)

    def test_mae(self):
        assert_frame_equal(mae_score(data), solution_MAE)

if __name__ == '__main__':
    unittest.main()
