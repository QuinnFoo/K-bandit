import pandas_datareader.data as web
import datetime
import numpy as np
import sys
import pandas_datareader as pdr
import string
import random
np.warnings.filterwarnings('ignore')

#set up the time
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2021, 1, 27)

#pick 10 stocks in random
seeds = string.digits

stocks=[]
index=web.DataReader("0001.HK", "yahoo", start, end).index[0]
while len(stocks)<10:
    random_str = []
    for i in range(4):
        random_str.append(random.choice(seeds))
    random_str.append('.HK')
    stock_inf=[]
    try: 
        stock_inf=web.DataReader("".join(random_str), "yahoo", start, end)
        if((stock_inf.ix[0,0]!=None) & (stock_inf.index[0]==index)):
               stocks.append("".join(random_str))
    except:
        continue

print(stocks)
#reload the all information of the stocks

stockList = web.DataReader(stocks, "yahoo", start, end)
print(stockList.head())
stockList.to_csv('./PandasNumpy.csv')
