import pandas as pd
import scipy
#Create a function for drowdown time series
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
    
    rets = me_m[["Lo 10", "Hi 10"]]
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
    
    