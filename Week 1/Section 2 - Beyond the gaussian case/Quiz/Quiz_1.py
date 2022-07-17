import edhec_risk_kit as erk
import numpy as np
import pandas as pd


#Question 1 - Find the annual return

#Get the data
data = erk.get_ffme_returns()
data = data["1999":"2015"]


n_months = len(data-1)

annualized_return= (data + 1).prod() ** (12/n_months) - 1
annualized_return


#Question 2 Find the annual volatility of Lo 20
annualized_vol = data.std()*np.sqrt(12)
annualized_vol


result_list = [annualized_vol, annualized_return]
comparison = pd.concat(result_list, axis = 1)
comparison.columns = ["Annual Vol", "Annual Return"]
comparison


#We have to find the max drawdown Lo_20

erk.drawdown(data["LargeCap"]["1999":"2015"])["Drawdown"].idxmin()

#Question 13 - Hedge Funds data 

data2 = erk.get_hfi_returns()

print(erk.kurtosis(data2["2000":]).sort_values())


