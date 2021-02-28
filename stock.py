import pandas_datareader.data as web
import datetime
import numpy as np
import sys
import pandas_datareader as pdr
import string
import random

# set up the time
start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2021, 1, 27)

# pick 10 stocks in random
seeds = string.digits

stocks = []
while len(stocks) < 10:
    random_str = []
    for i in range(4):
        random_str.append(random.choice(seeds))
    random_str.append('.HK')

    try:
        web.DataReader("".join(random_str), "yahoo", start, end)
        stocks.append("".join(random_str))
    except:
        continue

print(stocks)