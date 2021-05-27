import sys
sys.path.append('../')

import pandas as pd
from pandas.testing import assert_frame_equal
import unittest
from tools.algorithms import *


d = {'Predictor I': [2, 4, 6, 6], 'Predictor II': [5, 5, 6, 5], 'Predictor III': [10, 5, 8, 8]}
df = pd.DataFrame(data=d)

dtwo = {'Predictor I': [2, 4, 6, 6], 'Predictor II': [5, 5, 6, 5], 'Predictor III': [10, 5, 8, 8], 'Predictor IV': [2, 4, 6, 6]}
dftwo = pd.DataFrame(data=dtwo)

dthree = {'Predictor I': [2, 4, 6, 6], 'Predictor II': [5, 5, 6, 5]}
dfthree = pd.DataFrame(data=dthree)

d1 = {'Real Value': [6, 5, 6, 7]}
df1 = pd.DataFrame(data=d1)




solution_0 = {'System Disagreement': [3.555556, 0.444444, 0.888889, 1.3333333]}
solution_disagreement = pd.DataFrame(data=solution_0)

solution_0_1 = {'System Disagreement': [1.5, 0.5, 0.0, 0.5]}
solution_disagreement_0 = pd.DataFrame(data=solution_0_1)

solution_0_2 = {'System Disagreement': [3.375, 0.500, 0.750, 1.125]}
solution_disagreement_1 = pd.DataFrame(data=solution_0_2)


solution_1 = {0: [3.666667, 0.666667, 0.666667, 1.000000], 1: [2.666667, 0.333333, 0.666667, 1.333333],
              2: [4.333333, 0.333333, 1.333333, 1.666667]}
solution_predictor = pd.DataFrame(data=solution_1)

list_0 = [1, 3, 4, 100, [1, "test"], 23, [23, "test2"]]

solution_list_0 = [1, 3, 4, 100, 1, 23, 23]

list_1 = [2, 5, 10]
list_2 = [2, 6, 6]
value_2 = 6

solution_list_weights = [0.83333333333333334, 1.3333333333333333, 0.8333333333333334]
solution_list_weights_2 = [0.0, 1.5, 1.5]

solution_list_weights_focused = [0, 1, 0]
solution_list_weights_focused_2 = [0, 1, 1] 

solution_consolidation = [5.666666666666667, 4.722222222222222, 7.0, 5.5]
solution_consolidation_1 = [3.5, 4.8, 6.0, 5.5] # dfthree
solution_consolidation_2 = [4.75, 4.538461538461538, 6.666666666666666, 5.666666666666666]  # dftwo

solution_consolidation_memory = [5.666666666666667, 4.694444444444445, 6.7407407407407405, 6.111111111111111]
solution_consolidation_memory_1 = [3.5, 4.65, 6.0, 5.3] 
solution_consolidation_memory_2 = [4.75, 4.519230769230769, 6.542735042735042, 6.100961538461537]

solution_consolidation_focused = [4.75, 5.0, 7.0, 5.666666666666667]  # four
solution_consolidation_focused_1 = [5.666666666666667, 5.0, 7.0, 5.5]  # three
solution_consolidation_focused_2 = [3.5, 5.0, 6.0, 5.5]  

solution_consolidation_anchor = [6.12, 4.626288659793814, 7.005263157894737, 6.114705882352942]
solution_consolidation_anchor_1 = [3.65, 4.822695035460993, 6.08421052631579, 5.533333333333333] 
solution_consolidation_anchor_2 = [5.433333333333334, 4.524786324786326, 6.82, 6.091764705882354] 


solution_average_consolidation = [5.666666666666667, 4.666666666666667, 6.666666666666667, 6.333333333333333]

class Testing(unittest.TestCase):

    def test_disagreement_threePredictors(self):
        assert_frame_equal(disagreement(df), solution_disagreement)
    
    def test_disagreement_fourPredictors(self):
        assert_frame_equal(disagreement(dftwo), solution_disagreement_1)
    
    def test_disagreement_twoPredictors(self):
        assert_frame_equal(disagreement(dfthree), solution_disagreement_0)

    def test_predictor_score(self):
        assert_frame_equal(predictor_score(df), solution_predictor)

    def test_formatting(self):
        self.assertEqual(formatting(list_0), solution_list_0, 'Does not match solution')

    def test_weights_1(self):
        self.assertEqual(new_weights(list_1, value_2), solution_list_weights, 'Does not match solution')

    def test_weights_2(self):
        self.assertEqual(new_weights(list_2, value_2), solution_list_weights_2, 'Does not match solution')

    def test_weights_focused_1(self):
        self.assertEqual(new_weights_focused(list_1, value_2), solution_list_weights_focused, 'Does not match solution')

    def test_weights_focused_2(self):
        self.assertEqual(new_weights_focused(list_2, value_2), solution_list_weights_focused_2, 'Does not match solution')

    def test_consolidation(self):
        self.assertEqual(consolidated_predictions(df, df1), solution_consolidation, 'Does not match solution')

    def test_consolidation_twoPredictors(self):
        self.assertEqual(consolidated_predictions(dfthree, df1), solution_consolidation_1, 'Does not match solution')

    def test_consolidation_fourPredictors(self):
        self.assertEqual(consolidated_predictions(dftwo, df1), solution_consolidation_2, 'Does not match solution')

    def test_consolidation_memory(self):
        self.assertEqual(consolidated_predictions_memory(df, df1), solution_consolidation_memory, 'Does not match solution')

    def test_consolidation_memory_twoPredictors(self):
        self.assertEqual(consolidated_predictions_memory(dfthree, df1), solution_consolidation_memory_1, 'Does not match solution')

    def test_consolidation_memory_fourPredictors(self):
        self.assertEqual(consolidated_predictions_memory(dftwo, df1), solution_consolidation_memory_2, 'Does not match solution')
    
    def test_consolidation_focused(self):
        self.assertEqual(consolidated_predictions_focused(df, df1), solution_consolidation_focused_1, 'Does not match solution')

    def test_consolidation_focused_twoPredictors(self):
        self.assertEqual(consolidated_predictions_focused(dfthree, df1), solution_consolidation_focused_2, 'Does not match solution')

    def test_consolidation_focused_fourPredictors(self):
        self.assertEqual(consolidated_predictions_focused(dftwo, df1), solution_consolidation_focused, 'Does not match solution')

    def test_consolidation_anchor(self):
        self.assertEqual(consolidated_predictions_anchor(df, df1, 1.2), solution_consolidation_anchor, 'Does not match solution')

    def test_consolidation_anchor_twoPredictors(self):
        self.assertEqual(consolidated_predictions_anchor(dfthree, df1, 1.2), solution_consolidation_anchor_1, 'Does not match solution')

    def test_consolidation_anchor_fourPredictors(self):
        self.assertEqual(consolidated_predictions_anchor(dftwo, df1, 1.2), solution_consolidation_anchor_2, 'Does not match solution')

    def test_average_consolidation(self):
        self.assertEqual(average_consolidation(df), solution_average_consolidation, 'Does not match solution')


if __name__ == '__main__':
    unittest.main()
