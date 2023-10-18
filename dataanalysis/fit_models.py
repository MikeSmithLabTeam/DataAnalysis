import numpy as np

'''
To add a new fit model you need to provide a function that has the x and fit params listed
and returns the value f(x). Then you need to add to the dictionary a string which describes the equation
and the number of fit parameters as an int.
Optionally also provide a function to make a guess based on the data
of intitial parameter values. This should be named 'fitfunctionname' + '_guess'.
'''

'''
This dictionary provides the equation string and the number of fit parameters.
'''
fit_dict = {
            'linear': ('f(x) = a*x + b', 2),
            'quadratic': ('f(x) = a*x**2 + b*x + c', 3),
            'cubic': ('f(x) = a*x**3 + b*x**2 + c*x + d', 4),
            'exponential': ('f(x) = a*exp(b*x)', 2),
            'flipped_exponential': ('f(x) = a*(1 - exp(b*(x-c))+d)', 4),
            'double_flipped_exponential':('a*(1-c*np.exp(b*(x-f))-e*np.exp(d*(x-f)))+g', 7),
            'sin_cos': ('f(x) = asin(bx)+bcos(cx)+d', 4),
            'gaussian': ('f(x) = aexp(-(x-b)**2/(2c**2))', 3),
            'dbl_gaussian': ('f(x) = aexp(-(x-b)**2/(2c**2)) + dexp(-(x-e)**2/(2f**2))', 6),
            'triple_gaussian': ('f(x) = aexp(-(x-b)**2/(2c**2)) + dexp(-(x-e)**2/(2f**2)) + gexp(-(x-h)**2/(2j**2))', 9),
            'poisson': ('f(x)=a*(b**c)*exp(-b)/c!', 3),
            'axb':('f(x)=a(x)**b',2),
            'lorentzian':('f(x)=(c)*(a/2)**2/((x-b)**2 + (a/2)**2)', 3),
            'triple_lorentzian':('f(x)=(c)*(a/2)/((x-b)**2 + (a/2)**2) + (f)*(d/2)/((x-e)**2 + (d/2)**2) + (j)*(g/2)/((x-h)**2 + (g/2)**2)', 9),
            'dipole_fit':('f(x)=x/(a*sin(b+x))', 2),
           }

'''
Each fitting function has a function for the type of fit called fittype and optionally a function
which makes a guess of the initial fitting parameters called fittype_guess
'''
'''custom fits'''
def dipole_fit(x, a, b):
    b=b*np.pi/180
    return x/(a*np.sin(b+x))


'''
Polynomial functions
'''


def linear(x, a, b):
    return a*x + b

def linear_guess(x, y):
    b = 0
    a = (y.max()-y.min())/(x.max()-x.min())
    return [a, b]

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def axb(x,a,b):
    '''Simple power law'''
    return a*(x)**b

def axb_guess(x, y):
    a = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
    b = 0
    return [a, b]

'''
Exponential functions
'''


def exponential(x, a, b):
    return a*np.exp(b*x)


def flipped_exponential(x, a, b, c, d):
    return a*(1-np.exp(-b*(x-c))+d)


def double_flipped_exponential(x, a, b, c, d, e, f, g):
    return a*(1-c*np.exp(-b*(x-f))-e*np.exp(-d*(x-f))+g)



'''
Probability distributions
'''
def lorentzian(x, a, b, c):
    return (c)*(a/2)/((x-b)**2 + (a/2)**2)

def lorentzian_guess(x, y):
    b = x[np.argmax(y)]
    a = np.std(x)
    c = np.max(y)
    return [a, b, c]

def triple_lorentzian(x, a, b, c, d, e, f, g, h, j):
     return (c * (a / 2) / ((x - b)**2 + (a / 2)**2)) \
            + f * (d / 2) / ((x - e)**2 + (d / 2)**2) \
            + j * (g / 2) / ((x - h)**2 + (g / 2)**2)

def triple_lorentzian_guess(x, y):
    c = np.max(y) / 2
    f = np.max(y) / 2
    j = np.max(y / 2)
    b = np.mean(x)
    e = np.mean(x) + np.std(x)
    h = np.mean(x) - np.std(x)
    a = np.std(x) / 2
    d = np.std(x) / 2
    g = np.std(x) / 2
    return [a, b, c, d, e, f, g, h, j]

def gaussian(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))


def gaussian_guess(x, y):
    """
    Performs some simple calculations to guess initial params for gaussian fit.
    """
    a = np.max(y)
    b = np.mean(x)
    N = np.sum(y)
    c = 0.5*np.sqrt((1/N)*np.sum(y*(x-b)**2))
    return [a, b, c]

def area_gaussian(a,c):
    return np.sqrt(2*np.pi)*a*np.abs(c)

def dbl_gaussian(x,a,b,c,d,e,f):
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2))

def dbl_gaussian_guess(x, y):
    a = np.max(y)/2
    d = np.max(y)/2
    b = np.mean(x)+1/np.max(x)
    e = np.mean(x)-1/np.max(x)
    c = np.std(x)/2
    f = np.std(x)
    return [a, b, c, d, e, f]

def triple_gaussian(x,a,b,c,d,e,f,g,h,j):
    return a*np.exp(-(x-b)**2/(2*c**2)) + d*np.exp(-(x-e)**2/(2*f**2)) + g*np.exp(-(x-h)**2/(2*j**2))


def triple_gaussian_guess(x, y):
    a = np.max(y)/2
    d = np.max(y)/2
    g = np.max(y/2)
    b = np.mean(x)
    e = np.mean(x)+ np.std(x)
    h = np.mean(x)- np.std(x)
    c = np.std(x)/2
    f = np.std(x)/2
    j = np.std(x)/2

    return [a, b, c, d, e, f, g, h, j]

def lorentz_dblgaussian(x, a, b, c, d,e,f,g,h,j):
    return a*(c/2)**2/((x-b)**2 + (c/2)**2) + d*np.exp(-(x-e)**2/(2*f**2)) + g*np.exp(-(x-h)**2/(2*j**2))

def lorentz_dblgaussian_guess(x, y):
    a = np.max(y) / 2
    d = np.max(y) / 2
    g = np.max(y / 2)
    b = np.mean(x)
    e = np.mean(x) + np.std(x)
    h = np.mean(x) - np.std(x)
    c = np.std(x) / 2
    f = np.std(x) / 2
    j = np.std(x) / 2
    return [a,b,c,d,e,f,g,h,j]

'''
sin wave fitting is incredibly sensitive to phase c so use the following form
'''


def sin_cos(x, a, b, c, d):

    return a * np.sin(c * x) + b * np.cos(c * x) + d


def sin_cos_guess(x, y):
    A = np.std(y)/np.sqrt(2)
    D = np.mean(y)
    B = 0
    C = fft_power_spectrum(x, y)[2]
    return sin_const_convert([A, B, C, D], long=False)


def sin_const_convert(params, long=True):
    """
    There are two equivalent forms 1) Asin(CX + B) + D and 2) asin(cx) +  bcos(cx) + d. 
    This converts the constants between 1 and 2 if long == False and 2 and 1 if long == True  
    Maths is here:
    http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/trig/phase.html
    c and d or C and D remain unchanged
    """
    if long:
        print('Changing: a sin(cx) + b cos(cx) + d')
        print('to : A sin(CX + B) + D')
        a = params[0]
        b = params[1]
        
        params[1] = np.arctan2(b,a) #B
        params[0] = (a**2 + b**2)**0.5 #A
        
    else:
        print('Changing : A sin (CX + B) + D')
        print('to : a sin(cx) + b cos(cx) + d')
        
        A = params[0]
        B = params[1]
        
        params[1] = A*np.cos(B)
        params[0] = A*np.sin(B)
    
    return params


