from email import message
import hello as h
import edhec_risk_kit as erk
import pandas as pd





returns = erk.get_ffme_returns()




print(erk.drawdown(returns["SmallCap"])["Drawdown"].min())