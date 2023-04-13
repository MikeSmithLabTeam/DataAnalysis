'''
Exception defintions
'''


class DataLengthException(Exception):
    def __init__(self, x, y):
        len_x = np.shape(x)[0]
        len_y = np.shape(y)[0]
        if len_x != len_y:
            print('The arrays are different lengths')
            print(len_x)
            print(len_y)


class ParamNumberException(Exception):
    def __init__(self, fit_num_params, guess):
        len_guess = np.shape(guess)[0]
        print('fit_num_params is ', fit_num_params)
        print('params guess has only ', len_guess)


class FitTypeGuessException(Exception):
    def __init__(self, fittype):
        print(fittype, 'has not yet been defined for this function')
        print('You must define initial parameters manually')