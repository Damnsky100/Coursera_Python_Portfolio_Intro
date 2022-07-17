import pandas as pd
import edhec_risk_kit as erk
import scipy.stats
import numpy as np




hfi = erk.get_hfi_returns()

#Observe if they asset class as negative skewness by comparing the mean and the median
temp = pd.concat([hfi.mean(), hfi.median(), hfi.mean()>hfi.median()], axis = "columns")
temp.columns = ["Mean", "Median", "Bool"]
#print(temp)

#Calculate the skewness

temp2 = erk.skewness(hfi).sort_values()
#print(scipy.stats.skew(hfi))

#print(temp2)

#normal distribution test

normal_rets = np.random.normal(0, 0.15, size = (263, 1))
#print(erk.skewness(normal_rets))


#Calculate the kurtosis

temp3 = erk.kurtosis(normal_rets)
#print(temp3)

temp4 = erk.kurtosis(hfi).sort_values() #We could use scipy.stats.kurtosis --> gives you the excess kurtosis of a normal distribution (result - 3)
#print(temp4)


#Jacque Bera test --> to see if the series a return is normally distributed



hfi.aggregate(erk.is_normal) #Aggregate permet d'infliger la fonction à toute les colonnes


ffme =erk.get_ffme_returns()
print(erk.skewness(ffme))
print(erk.kurtosis(ffme))
print(ffme.aggregate(erk.is_normal))