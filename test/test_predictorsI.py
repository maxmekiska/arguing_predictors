import pandas as pd
import numpy as np
import unittest

def sequence_prep(input_sequence, steps_past: int, steps_future: int):
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
    X = []
    y = []
    if length <= steps_past:
        raise ValueError('Input sequence is equal to or shorter than steps to look backwards')
    if steps_future <= 0:
        raise ValueError('Steps in the future need to be bigger than 0')

    for i in range(length):
        last = i + steps_past
        if last > length - steps_future:
            break
        X.append(input_sequence[i:last])
        y.append(input_sequence[last:last + steps_future])
    y = np.array(y)
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

test_price = np.array([28.12999916, 27.79999924, 27.79999924, 27.80999947, 27.48999977,
       27.70999908, 27.11000061])

solution_X = np.array([[[28.12999916],
        [27.79999924]],
       [[27.79999924],
        [27.79999924]],
       [[27.79999924],
        [27.80999947]],
       [[27.80999947],
        [27.48999977]],
       [[27.48999977],
        [27.70999908]]])

solution_y = np.array([[27.79999924],
       [27.80999947],
       [27.48999977],
       [27.70999908],
       [27.11000061]])

class Testing(unittest.TestCase):

    def test_sequence_pred_X(self):
        X, _ = sequence_prep(test_price, 2, 1)
        np.testing.assert_array_equal(X, solution_X, 'not eaqual')

    def test_seuence_pred_y(self):
        _, y = sequence_prep(test_price, 2, 1)
        np.testing.assert_array_equal(y, solution_y, 'not equal')

if __name__ == '__main__':
    unittest.main()    
