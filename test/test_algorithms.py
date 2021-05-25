import sys
sys.path.append('../')

import pandas as pd
from pandas.util.testing import assert_frame_equal
import unittest
from tools.algorithms import *
from test_variables_algorithms import *


class Testing(unittest.TestCase):

    def test_disagreement(self):
        assert_frame_equal(disagreement(df), solution_disagreement)

    def test_predictor_score(self):
        assert_frame_equal(predictor_score(df), solution_predictor)

    def test_formatting(self):
        self.assertEqual(formatting(list_0), solution_list_0, 'Does not match solution')

    def test_weights(self):
        self.assertEqual(new_weights(list_1, value_2), solution_list_weights, 'Does not match solution')

    def test_consolidation(self):
        self.assertEqual(consolidated_predictions(df, df1), solution_consolidation, 'Does not match solution')

    def test_consolidation_memory(self):
        self.assertEqual(consolidated_predictions_memory(df, df1), solution_consolidation_memory, 'Does not match solution')

    def test_consolidation_anchor(self):
        self.assertEqual(consolidated_predictions_anchor(df, df1, 1.2), solution_consolidation_anchor, 'Does not match solution')
    def test_average_consolidation(self):
        self.assertEqual(average_consolidation(df), solution_average_consolidation, 'Does not match solution')


if __name__ == '__main__':
    unittest.main()
