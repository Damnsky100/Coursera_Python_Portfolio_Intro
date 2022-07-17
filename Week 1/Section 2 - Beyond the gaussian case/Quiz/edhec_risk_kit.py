import pandas as pd
import scipy
import numpy as np
#Create a function for drawdown time series
def drawdown(return_series : pd.Series):
    """
    Takes time series of asset returns
    Computes and returns a Dataframe that contains:
    The wealth index
    the previous peak
    percentage drawdowns
    """
    wealth_index = 1000 * (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })
    
    
def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the top and Bottom Decile by Marketcap
    """
    me_m = pd.read_csv("/Users/sebastiencaron/Desktop/Portfolio Construction with Python/Excel Data/Portfolios_Formed_on_ME_monthly_EW.csv", header=0, index_col=0, na_values=-99.99)
    
    rets = me_m[["Lo 20", "Hi 20"]]
    rets.columns = ["SmallCap", "LargeCap"]
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format = "%Y%m").to_period("M")
    return rets


def get_hfi_returns():
    """
    Load and Format de EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("/Users/sebastiencaron/Desktop/Portfolio Construction with Python/Excel Data/edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")
    return hfi



def semideviation(r):
    """
    Returns the semideviation aka negative semideviation or r
    r must be a Series or a Dataframe
    """
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Return a float or a series
    """
    demeaned = r-r.mean()
    #use the population standard deviation, so we set dfo=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned**3).mean()
    return exp / sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Return a float or a series
    """
    demeaned = r-r.mean()
    #use the population standard deviation, so we set dfo=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned**4).mean()
    return exp / sigma_r**4


def is_normal(r, level = 0.01): #0.01 is a default value
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not 
    Test is applied at the 1 percent level of by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level
    

def var_historic(r, level =5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else: 
        raise TypeError("Expected r to be a Series or DataFrame")
    
    
from scipy.stats import norm
def var_gaussian(r, level=5, modified =False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    #Compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)#Percent point function, we want the z score
    if modified:
        #modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 -5*z)*(s**2)/36 
             
             )
        
    return -(r.mean()+ z * r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Compute the Conditionnal VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
