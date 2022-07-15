import numpy as np
import pandas as pd


prices_a = np.array([8.70, 8.91, 8.71]) #Numpy array permet de diviser deux array afin d'avoir le rendement

def return_1 (actif, t):
    return (actif[t] / actif[t-1]) - 1

print(return_1(prices_a, 1))

def return_all(actif):
   return (actif[len(actif)-1] / actif[0] - 1)

print(return_all(prices_a))

print((prices_a[1:]/prices_a[:-1])-1)

prices = pd.DataFrame({
    "Blue": [8.70, 8.91, 8.71, 8.43, 8.73],
    "Orange": [10.66, 11.08, 10.71, 11.59, 12.11]
    
                }) 
                       
print(prices)