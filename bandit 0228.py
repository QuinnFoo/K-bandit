#!/usr/bin/env python
# coding: utf-8




# %load HW1.py
#Some intuition:  we need:
# 1. collect and process the stock data, distribute certain amount of money (maybe $100) everyday
# 2. DEFINE State: the current stock amount? current price
# 3. DEFINE Action: sell, buy, hold with amount Weight[-1, 1]
# 4. DEFINE Reward(a function with step t): let's try maximum the return first
# 5. DEFINE Agent: take one algorithm(epsilon greedy ?), value estimation ,choose action based on State
# 6. Run Training (like 1000 epochs)
##################################################################################################################


#need to install database ! pip install pandas_datareader





import pandas_datareader.data as web
import datetime
import numpy as np
import sys
import pandas_datareader as pdr





import string
import random

#set up the time
start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2021, 1, 27)

#pick 10 stocks in random
seeds = string.digits

stocks=[]
while len(stocks)<10:
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





#reload the all information of the stocks

stockList = web.DataReader(stocks, "yahoo", start, end)
print(stockList.head())
stockList.to_csv('./PandasNumpy.csv')





import math
import numpy as np
import pandas as pd

#steps: t
t=1000

#TODO: stockList: a dataframe of stocks for us to invest
stockList = pd.DataFrame()
actionCount = len(stock List.colomns)

#stock: time(date), open, close,
def dailyReturn(stock, t, money=100):
    open, close = stockList[stock][t]
    shares = money/open
    return shares*close - money

#we call rewardFun() to compute reward each step and action for us.
#I don't use dailyReturn as rewardFun is that maybe we can transfer it to other form, like dropdown, later.
def rewardFun(action, t):
    return dailyReturn(action, t)





############################## ALGORITHM PART #############################################
#actionCount is the total number of action(0,1,2..)

def eps_Greedy(epsilon,rewardFun,actionCount):
    reward = [0.0]*actionCount
    Qta = [0.0]*actionCount
    NumAction=[1] * actionCount
    p=np.random.rand(actionCount)

    for i in range(1,t):
        if p <= epsilon:
            Action = np.random.choice(actionCount)
        else:
            Action = reward.index(max(Qta))

        #update
        NumAction[Action] += 1
        reward[Action] += rewardFun(i,Action)
        Qta[Action] += reward[Action]/NumAction[Action]

        yield Action, reward[Action]

#make epsilon decay: epsilon=1/(1+t/actionCount)
def decayEps_Greedy(rewardFun, actionCount):
    reward = [0.0]*actionCount
    Qta = [0.0]*actionCount
    NumAction=[1] * actionCount
    p=np.random.rand(actionCount)

    for i in range(1,t):
        if p <= 1/(1+i/actionCount):
            Action = np.random.choice(actionCount)
        else:
            Action = reward.index(max(Qta))

        #update
        NumAction[Action] += 1
        reward[Action] += rewardFun(i,Action)
        Qta[Action] += reward[Action]/NumAction[Action]

        yield Action, reward[Action]

def UCB(rewardFun, actionCount, c=2):
    #for each action, initialize:
    #NumAction(Nt(a)) is the times each action has been done
    NumAction= [1] * actionCount
    Qta= [0.0] * actionCount
    reward= [0.0] * actionCount

    #ucbF is that [] in max formula to chose At, we want the Action At with max ucbF
    #initialize ucbF with 0
    ucbF= [0.0] * actionCount

    for i in range(1,t):
        #count ucbF for each action every step
        for j in range(NumAction):
            ucbF[j]= Qta[j] + c * math.sqrt(math.log(i)/NumAction[j])

        #choose the Action index whose ucbF is max
        Action= ucbF.index(max(ucbF))
        NumAction[Action] += 1
        reward[Action] += rewardFun(i,Action)
        Qta[Action] += reward[Action]/NumAction[Action]

        yield Action, reward[Action]






############################## ALGORITHM PART #############################################

#TODO: visualize the reward
#git test

