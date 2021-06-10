from numpy import array
import unittest
import numpy as np

def sequence_prep(input_sequence: array, sub_seq: int, steps_past: int, steps_future: int) -> array:
    '''Prepares data input into X and y sequences. Lenght of the X sequence is dertermined by steps_past while the length of y is determined by steps_future. In detail, the predictor looks at sequence X and predicts sequence y.

        Parameters:
            input_sequence (array): Sequence that contains time series in array format
            sub_seq (int): Further division of given steps a predictor will look backward.
            steps_past (int): Steps the predictor will look backward
            steps_future (int): Steps the predictor will look forward

        Returns:
            X (array): Array containing all looking back sequences
            y (array): Array containing all looking forward sequences
            modified_back (int): Modified looking back sequence length
    '''
    length = len(input_sequence)
    if length == 0:
        return (0, 0, steps_past // sub_seq)
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
    y = array(y)
    X = array(X)
    modified_back = X.shape[1]//sub_seq
    X = X.reshape((X.shape[0], sub_seq, modified_back, 1))
    return X, y, modified_back # special treatment to account for sub sequence division

test_price = np.array([28.12999916, 27.79999924, 27.79999924, 27.80999947, 27.48999977,
       27.70999908, 27.11000061])

solution_X = np.array([[[[28.12999916]],
        [[27.79999924]]],
       [[[27.79999924]],
        [[27.79999924]]],
       [[[27.79999924]],
        [[27.80999947]]],
       [[[27.80999947]],
        [[27.48999977]]]])

solution_y = np.array([[27.79999924, 27.80999947],
       [27.80999947, 27.48999977],
       [27.48999977, 27.70999908],
       [27.70999908, 27.11000061]])

solution_modified_back = 1

class Testing(unittest.TestCase):

    def test_sequence_pred_X(self):
        X, _, _ = sequence_prep(test_price, 2, 2, 2)
        np.testing.assert_array_equal(X, solution_X, 'not eaqual')

    def test_sequence_pred_y(self):
        _, y, _ = sequence_prep(test_price, 2, 2, 2)
        np.testing.assert_array_equal(y, solution_y, 'not equal')

    def test_sequence_pred_m(self):
        _, _, m = sequence_prep(test_price, 2, 2, 2)
        np.testing.assert_array_equal(m, solution_modified_back, 'not equal')

if __name__ == '__main__':
    unittest.main()
