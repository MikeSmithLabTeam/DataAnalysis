import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from select_data import get_pts
from fit_models import *
from custom_exceptions import DataLengthException, ParamNumberException, FitTypeGuessException
    

class Fit:
    """
    Generic fit object. Allows simple fitting of various functional forms together with viewing data
    and selecting appropriate values. Also provides simple statistics.
    
    Inputs:
    fit_type = function name that specifies the type of fit of interest. These are specified in the
                fit_dict and have a matching named function. The corresponding function with the same name 
                returns the value of the function for a specific set of parameter values.
    x,y = 1d numpy array data to be fitted of the form y = f(x)
    series = pandas series can be supplied in place of x,y data. It will plot data against index values
    
    guess = starting values for the fit -tuple with length = number of parameters
    lower = lower bounds on fit (optional)
    upper = upper bounds on fit (optional)
            These maybe a value, np.inf, -np.inf, None, None sets to +/- infinity, 'Fixed' sets the values to the guess values +/- 0.001%
    logic = a numpy array of length = x and y with True or False depending on whether data is to be included
    
    methods:
    __init__()  -     initialises with fit_type
    add_fit_data()  - can be used to add data or update data. reset_filter will
                      remove any logical filters that have been added. If the data is different
                      length this will autoreset regardless printing a message.
    add_params()  -   add fit parameters. lower and upper bounds are optional and will be set to +/- np.inf
                      if not supplied. 
    _guess_params()  - For some fitting functions a method has been implemented to guess intial parameter values
    _replace_none_fixed() - Internal method to replace None and 'Fixed' values in fit limits with appropriate substitutes
    add_filter()  -       Takes a logical numpy array with True or False to indicate if data should be included in fit. Also supports
                        graphical selection see below.
    fit()             Fits the data and returns (fit_params,fit_x, fit_y), optional parameter
                      errors which calculates errors on fit parameters using fit_errors(). This
                      may be slow especially for large datasets and hence is set to False by default. interpolation_factor
                      is an optional argument which returns the fit with the mean data spacing either increased or decreased
                      by the value stated.
    fit_errors()      Calculates the confidence interval on the fits. It estimates the
                      noise in the data based on the residuals. It creates some versions of the 
                      data with gaussian noise of same size as stdev of residuals and then
                      fits numfits times to these. The variance in these fits is then used
                      to calculate the ci on the fit parameters.
    plot_fit()        Plots fit to screen or file depending on settings
    stats()           provides simple statistics on the data and filtered data sets.
    
    Outputs:
    fit_params        A list of the parameters used to fit the data. Definitions are provided in the dict type fit_dict['*Fit Type*']
    fx             Returns the x values of the fit
    fy             Returns the values of the fit at each point

    Example Usage:
        
        xdata = np.arange(1000)
        a = 1
        b = 2
        y_linear = linear(xdata, a, b)

        linear_fit = Fit('linear', x=xdata, y=y_quadratic)
        linear_fit.add_params([4, 0])
        #Filter with logical array
        logic = xdata < 200
        linear_fit.add_filter(logic)
        #Filter by graphical selection
        linear_fit.add_filter()
        linear_fit.fit()
        linear_fit.plot_fit(show=True)

    """
    
    def __init__(self, fit_type,x=None,y=None, xlabel='x',ylabel='y',title='title'):
        self.filename = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.fit_type = fit_type
        self.fit_string, self._num_fit_params = fit_dict[fit_type]

        if x is not None:
            self.add_fit_data(x=x,y=y)

    def add_fit_data(self, x=None, y=None, series=None, reset_filter=True):
        if series is not None:
            x = series.index
            y = series.values
        self.x = x
        self.y = y
        
        if np.shape(x)[0] != np.shape(y)[0]:
            print('x', 'y')
            raise DataLengthException(x, y)
        if np.shape(x) != np.shape(self.x):
            reset_filter = True
            print('Data different length to current data, resetting logical filter')

        if reset_filter:
            self.add_filter(logic=np.ones(np.shape(x), dtype=bool))
    
    def add_params(self, guess=None, lower=None, upper=None):
        """
        Add fitting parameters. In addition to values the following special vals are
        supported. None, 'Fixed', np.inf, -np.inf.
        'Fixed' is only valid for starting value.

        Args:
            guess (Tuple of params, optional): Starting Value. Defaults to None.
            lower (Tuple of params, optional): Lower Bound. Defaults to None.
            upper (Tuple of params, optional): Upper Bound. Defaults to None.

        Raises:
            ParamNumberException: If number params is not appropriate for fit model
        """
        if guess is None:
            guess = self._guess_params()
        _num_params = np.shape(guess)[0]
        if _num_params != self._num_fit_params:
            raise ParamNumberException(guess)
        self._params = guess
        self._lower=lower
        self._upper=upper
        self._replace_none_fixed()
        print(self._params)

    def _guess_params(self):
        print(globals()[self.fit_type])
        try:
            guess = globals()[self.fit_type + '_guess'](self.fx, self.fy)
            self._params = guess
        except:
            raise FitTypeGuessException(self.fit_type)
        return guess
    
    def _replace_none_fixed(self,nudge = 0.001):
        if self._lower is None:
            self._lower = [-np.inf]*self._num_fit_params
        if self._upper is None:
            self._upper = [np.inf]*self._num_fit_params
    
        for index, item in enumerate(self._lower):
            if item is None:
                self._lower[index] = -np.inf
            elif (item == 'Fixed') and (self._params[index] < 0):
                self._lower[index] = self._params[index]*(1.0 + nudge)
            elif (item == 'Fixed') and (self._params[index] > 0):
                self._lower[index] = self._params[index]*(1.0 - nudge)
            elif (item == 'Fixed') and (self._params[index] == 0):
                self._lower[index] = -nudge
    
            for index, item in enumerate(self._upper):
                if item is None:
                    self._upper[index] = np.inf
                elif (item == 'Fixed') and (self._params[index] > 0):
                    self._upper[index] = self._params[index]*(1.0 + nudge)
                elif (item == 'Fixed') and (self._params[index] < 0):
                    self._upper[index] = self._params[index]*(1.0 - nudge)
                elif (item == 'Fixed') and (self._params[index] == 0):
                    self._upper[index] = nudge

    def add_filter(self, logic=None, selector_type='Rectangle'):
        """
        add_filter() allows you to be selective about which data
        points to include in the fit. You can do this by sending 
        a boolean array of length = to the data or by graphically
        selecting the data.

        Args:
            logic (bool array or None): boolean array of same length as data or None to start graphical selection
            type (str): Type of selector to use in graphical input. Options are 'Rectangle','Ellipse' or 'Lasso'. Defaults to 'Rectangle'.

        Raises:
            DataLengthException: If logic is not same length as data
        """
        if logic is None:
            self.plot_fit(fit=False, show=False)
            pt_indices, pt_values = get_pts(self.plot_ax, self.pts_handle, selector_type=selector_type)
            logic=np.zeros(np.shape(self.x),dtype=bool)
            logic[pt_indices]=True
        else:
            if type(logic) == type(pd.Series(dtype='float64')):
                logic = logic.values
            len_logic = np.shape(logic)[0]
            if len_logic != np.shape(self.x)[0]:
                print('x', 'logic')
                raise DataLengthException(self.x, logic)
        
        self.logic = logic
        self.fx = self.x[logic]
        self.fy = self.y[logic]

    def fit(self, interpolation_factor=1.0, errors=False):
        fit_output = optimize.curve_fit(globals()[self.fit_type],
                                        self.fx,
                                        self.fy,
                                        p0=self._params,
                                        bounds=(self._lower, self._upper))
        self.fit_params = fit_output[0]

        if errors:
            self.fit_errors()
        else:
            self.fit_param_errors = [np.nan]*int(fit_dict[self.fit_type][1])

        if interpolation_factor != 1.0:
            self.stats()
            original_step = (self.xdata_max - self.xdata_min) / self.xdata_length
            interpolation_step_size = interpolation_factor * original_step
            self.fit_x = np.arange(
                                   self.xdata_min,
                                   self.xdata_max,
                                   interpolation_step_size
                                  )
        else:
            self.fit_x = self.fx

        self.fit_y = globals()[self.fit_type](self.fit_x, *self.fit_params)
        print('\nFit : ', fit_dict[self.fit_type])
        print('Fit params : (param, lower, upper, ci) ')
        letters = [chr(c) for c in range(ord('a'),ord('z')+1)]
        for index,param in enumerate(self.fit_params):
            print(letters[index],': (', param, self._lower[index], self._upper[index], self.fit_param_errors[index], ')')
        
        return self.fit_params, self.fit_x, self.fit_y

    def fit_errors(self,numfits=100):
        error_func = lambda p, x, y: globals()[self.fit_type](x, *p) - y
        self.fit_residuals = error_func(self.fit_params, self.fx, self.fy)
        sigma_res = np.std(self.fit_residuals)
        
        ps = []
        for i in range(int(numfits)):
            randomDelta = np.random.normal(0., sigma_res, len(self.fy))
            randomdataY = self.fy + randomDelta

            randomfit, randomcov = optimize.leastsq(error_func, self.fit_params, args=(self.fx, randomdataY), full_output=False)
            ps.append(randomfit) 

        ps = np.array(ps)
        mean_pfit = np.mean(ps, 0)

        pfit_bootstrap = mean_pfit
        self.fit_param_errors = np.std(ps, 0)        

    def _plot_limits(self, data_array, lower=True):
        '''internal method to control axes limits correctly'''
        if lower == True:
            lim = np.min(data_array)
            upordown = -1
        else:
            lim = np.max(data_array)
            upordown = 1
        if lim < 0:
            lim=lim *(1 - upordown*0.1)
        else:
            lim = lim * (1 + upordown * 0.1)
        return lim

    def plot_fit(self, filename=None, residuals=False, fit=True, title=None, xlabel=None, ylabel=None, show=True, save=False):
        if residuals:
            fig, axes = plt.subplots(nrows=2,ncols=1,sharex=True)
            self.plot_ax, res_ax = axes
        else:
            fig, self.plot_ax = plt.subplots()

        if self.filename is None and filename is None:
            filename = ' '
        elif filename is not None:
            self.filename = filename
        if self.xlabel is None and xlabel is None:
            self.xlabel=''
        elif xlabel is not None:
            self.xlabel = xlabel
        if self.ylabel is None and ylabel is None:
            self.ylabel=''
        elif ylabel is not None:
            self.ylabel = ylabel
        if self.title is None and title is None:
            self.title=''
        elif title is not None:
            self.title = title

        self.pts_handle = self.plot_ax.scatter(self.x, self.y, marker='x', c='r')
        if fit:
            self.plot_ax.scatter(self.fx, self.fy, marker='x', c='b')
            self.plot_ax.plot(self.fit_x, self.fit_y, 'g-')
        self.plot_ax.set_xlabel(self.xlabel)
        self.plot_ax.set_ylabel(self.ylabel)
        #self.plot_ax.configure_yaxis(ylim=(self._plot_limits(self.y,lower=True),self._plot_limits(self.y,lower=False)),ylabel=self.ylabel)
        self.plot_ax.set_title(self.title)

        if residuals:
            res_ax.scatter(self.fx, self.fit_residuals, 'rx')
            res_ax.set_ylabel('Residuals')

        if save:
            plt.save_figure(filename)
        if show:
           plt.show()
        return

    def stats(self, show_stats=True):
        self.ydata_max = np.max(self.y)
        self.ydata_min = np.min(self.y)
        self.ydata_mean = np.mean(self.y)
        self.ydata_median = np.median(self.y)
        self.ydata_std = np.std(self.y)
        self.xdata_max = np.max(self.x)
        self.xdata_min = np.min(self.x)
        self.xdata_length = np.shape(self.x)[0]
        
        self.fydata_mean = np.mean(self.fy)
        self.fydata_std = np.std(self.fy)
        self.fydata_median = np.median(self.fy)
        self.fydata_max = np.max(self.fy)
        self.fydata_min = np.min(self.fy)
        self.fxdata_max = np.max(self.fx)
        self.fxdata_min = np.min(self.fx)
        self.fxdata_length = np.shape(self.fx)[0]
        
        if show_stats:
            print('ydata:')
            print('mean - ', self.ydata_mean)
            print('std - ', self.ydata_std)
            print('median - ', self.ydata_median)
            print('min - ', self.ydata_min)
            print('max - ', self.ydata_max)
            print('xdata:')
            print('min - ', self.xdata_min)
            print('max - ', self.xdata_max)
            print('data length - ', self.xdata_length)
            print('')
            print('ydata filtered:')
            print('mean - ', self.fydata_mean)
            print('std - ', self.fydata_std)
            print('median - ', self.fydata_median)
            print('min - ', self.fydata_min)
            print('max - ', self.fydata_max)
            print('xdata filtered:')
            print('min - ', self.fxdata_min)
            print('max - ', self.fxdata_max)
            print('data length - ', self.fxdata_length)
            



if __name__ == '__main__':
    '''This contains all the unit tests'''
    xdata = np.arange(1000)
    a = 1
    b = 2
    y_linear = linear(xdata, a, b)
    
    linear_fit = Fit('linear', x=xdata, y=y_linear)
    
    linear_fit.add_params([4, 0])
    #Filter with logical array
    #logic = xdata < 200
    #linear_fit.add_filter(logic)
    #Filter by graphical selection
    linear_fit.add_filter()
    linear_fit.fit()
    linear_fit.plot_fit(show=True)
    
    
    
                
                
                
    
                         
