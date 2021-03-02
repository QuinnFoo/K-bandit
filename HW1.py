##########################################################################################
# Idea: K-arms are stocks.
# Action is picking a certain stock to invest everyday.
# Invest method: buy at open, sell at close.
# reward: daily return.
##########################################################################################
import math
import numpy as np
import pandas as pd


df = pd.read_csv('PandasNumpy.csv', index_col=0, header=[0, 1])
index = df.columns.get_level_values(1)[0:10]

openList = df.iloc[:, 30:40]
openList.columns = index

closeList = df.iloc[:, 10:20]
closeList.columns = index

# stock name list, also the choice of action
ActionCount = len(openList.columns)

t = len(openList)


# stock: time(date), open, close,
def rewardReturn(t, stock, money=100):
    open = openList.iloc[t, stock]
    close = closeList.iloc[t, stock]
    shares = money / open
    return (shares * close - money)


############################## ALGORITHM PART #############################################
# actionCount is the total number of action(0,1,2..)

def eps_Greedy(eps, step):
    p = np.random.rand()

    if p <= eps:
        action = np.random.choice(ActionCount)
    else:
        action = Qta.index(max(Qta))

    # update
    reward[action] += rewardReturn(step, action)
    Qta[action] += reward[action] / NumAction[action]
    NumAction[action] += 1
    return rewardReturn(step, action)


# make epsilon decay: epsilon=1/(1+t/actionCount)
def decayEps_Greedy(step):
    p = np.random.rand()

    if p <= 1 / (1 + (step+1) / ActionCount):
        action = np.random.choice(ActionCount)
    else:
        action = Qta.index(max(Qta))

    # update
    NumAction[action] += 1
    reward[action] += rewardReturn(step, action)
    Qta[action] += reward[action] / NumAction[action]

    return rewardReturn(step, action)


def UCB(step, c=2):
    # NumAction(Nt(a)) is the times each action has been done
    # ucbF is that [] in max formula to chose At, we want the Action At with max ucbF
    # initialize ucbF with 0

    # count ucbF for each action every step
    for j in range(ActionCount):
        ucbF[j] = Qta[j] + c * (math.sqrt((math.log(step+1))/ NumAction[j]))

        # choose the Action index whose ucbF is max
    action = ucbF.index(max(ucbF))

    reward[action] += rewardReturn(step, action)
    Qta[action] += reward[action] / NumAction[action]
    NumAction[action] += 1
    return rewardReturn(step, action)


############################## ALGORITHM PART #############################################
# initialization
#they are every step return
traceReturn1 = []
traceReturn2 = []
traceReturn3 = []

############################epsilon greedy result#################
epsilon = 0.9
reward = [0.0] * ActionCount
Qta = [0.0] * ActionCount
NumAction = [1] * ActionCount

for i in range(t):
    r = eps_Greedy(epsilon, i)
    traceReturn1.append(r)

############################greedy decay result###################
reward = [0.0] * ActionCount
Qta = [0.0] * ActionCount
NumAction = [1] * ActionCount

for i in range(t):
    r = decayEps_Greedy(i)
    traceReturn2.append(r)

###########################UCB result##################
reward = [0.0] * ActionCount
Qta = [0.0] * ActionCount
NumAction = [1] * ActionCount
ucbF = [0.0] * ActionCount

for i in range(t):
    r = UCB(i, c=2)
    traceReturn3.append(r)
