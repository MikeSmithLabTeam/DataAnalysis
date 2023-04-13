import numpy as np
from pytest import raises

from DataAnalysis.fitting import Fit
from DataAnalysis.custom_exceptions import DataLengthException, ParamNumberException, FitTypeGuessException

def test_add_data_raises_DataLengthException():
    with raises(DataLengthException):
        Fit('linear', x=np.array([1,2,3]), y=np.array([1,2,3,4]))