import pandas as pd
import edhec_risk_kit as erk

hfi = erk.get_hfi_returns() #Load data hedge fund



#Semi-deviation, we compute deviations for negative return
hfi.std(ddof=0) #we have the population data, so no freedom of degree


hfi[hfi<0].std(ddof=0)

# VaR and Cvar --> Value at Risk
"""
    1. Historic VaR
    2. Parametric VaR Gaussian
    3. Modified VaR - Cornish-Fisher VaR
"""
#  1. Historic VaR (Look at the 5% worst result)
import numpy as np
np.percentile(hfi, 5, axis=0) #the style output is not beautiful, lets write a function

    
erk.var_historic(hfi) #See the function in erk_risk_kit
    
  
  
#  2. Parametric VaR Gaussian  
print(erk.var_gaussian(hfi))


#3. Modified VaR - Cornish-Fisher VaR

"""
adjust the skewness and kurtosis
"""

var_list = [erk.var_gaussian(hfi), erk.var_gaussian(hfi, modified = True), erk.var_historic(hfi)]
comparison = pd.concat(var_list, axis=1)
comparison.columns = ["Gaussian", "Cornish-Fisher", "Historic"]

print(comparison)
print(comparison.plot.bar(title="EDHEC Hedge Fund Indices VaR"))



#Beyond VaR aka CVaR


#return that are worst than the var

print(erk.cvar_historic(hfi))