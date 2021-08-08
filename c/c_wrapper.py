import os
import ctypes

# get current directory to load shared C library "lib.so" - to build the shared library, use command: gcc -fPIC -shared -0 lib.so average.c correcting.c disagreement.c
current_dir = os.getcwd()
full_path = current_dir + "/lib.so"

_lib = ctypes.CDLL(full_path)

# Define arguments and return types for disagreement function
_lib.disagreement.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.c_int)
_lib.disagreement.restype = ctypes.c_double

# Define arguments and return types for consensus function
_lib.consensus.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int)
_lib.consensus.restype = ctypes.c_double

# wrapper for disagreement function
def disagreement(predictor_forecasts: list) -> float:
    '''Wrapper function to apply the disagreement C function within a python program.

        Parameters:
            predictor_forecasts (list): List containing all individual predictor forecasts for at time t.
        
        Returns:
            disagreement (float): System disagreement value.
    '''
    length = len(predictor_forecasts)
    array_type = ctypes.c_double * length
    disagreement = _lib.disagreement(array_type(*predictor_forecasts), ctypes.c_int(length))
    return float(disagreement)

# wrapper for consensus function
def consensus(old_pred: list, new_pred: list, real: float) -> float:
    '''Wrapper function to apply the correcting consensus C function within a python program.

        Parameters:
            old_pred (list): Individual predictor predictions at time t-1.
            new_pred (list): Individual predictor predictions at time t0.
            real (float): Real value at time t0 corresponding to predictions from t-1.
        
        Returns:
            consensus (float): System consensus value.
    '''
    length = len(old_pred)
    array_type = ctypes.c_double * length
    result = _lib.consensus(array_type(*old_pred), array_type(*new_pred), ctypes.c_double(real), ctypes.c_int(length))
    return float(result)

