##########################################################################################
# Idea: K-arms are stocks.
# Action is picking a certain stock to invest everyday.
# Do one action each day.Number of step is equal to time span.
# Invest method: use $100 each day, pick one stock, buy at open, sell at close.
# reward: daily return.
##########################################################################################
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#prepare for data,select price on open time and close time from stocks' information
df = pd.read_csv('PandasNumpy.csv', index_col=0, header=[0, 1])
index = df.columns.get_level_values(1)[0:10]

openList = df.iloc[:, 30:40]
openList.columns = index

closeList = df.iloc[:, 10:20]
closeList.columns = index

#we will pick one stock each step,so choosing a stock is an action 
#    ActionCount: length of the stocks' list columns is the choice of action
#    stepCount: we will make action each timestamp,so length of the stocklist is the total number of steps
ActionCount = len(openList.columns)
stepCount = len(openList)


# buy a stock on open time and sell on close time,then have the rewardreturn
def rewardReturn(t, stock, money=100):
    openprice = openList.iloc[t, stock]
    closeprice = closeList.iloc[t, stock]
    shares = int(money / openprice)
    return round((shares * (closeprice - openprice)),2)



#initialize reward,Qta,NumAction and cumulative
#    reward:a list of reward of each action prior to next step
#    Qta:Q(t+1) a list of  weighted average reward of each action prior to next step
#    NumAction:N(t+1) a list of the times each action has been done
#    culmulative: a list of cumulative return of method by time t
#Above variable will be update after action finished on time t

def initial():
    global reward 
    reward = [0.0] * ActionCount
    global Qta
    Qta = [0.0] * ActionCount
    global NumAction 
    NumAction = [0] * ActionCount
    global cumulative 
    cumulative = 0


############################## ALGORITHM PART #############################################

def eps_Greedy(eps, step):
    # eps_Greedy use epsilon_Greedy method to balance exploration and exploitation
    p = np.random.rand()
    
    #get an action for this step
    if p <= eps:
        action = np.random.choice(ActionCount)
    else:
        action = Qta.index(max(Qta))

    # update
    temp=rewardReturn(step, action)
    NumAction[action] += 1
    reward[action] += temp
    Qta[action] = reward[action] / NumAction[action]
    
    #return reward of this step
    return temp



def decayEps_Greedy(step):
    #As we learn for a long time, we can make more educated decisions and explore less
    #To increase times of exploitation over time,make epsilon decay: epsilon=1/(1+t/actionCount)
    p = np.random.rand()
    
    #get an action for this step
    if p <= 1 / (1 + (step+1) / ActionCount):
        action = np.random.choice(ActionCount)
    else:
        action = Qta.index(max(Qta))

    # update
    temp=rewardReturn(step, action)
    NumAction[action] += 1
    reward[action] += temp
    Qta[action] = reward[action] / NumAction[action]
    
    #return reward of this step
    return temp



def UCB(step, c=2):
    # NumAction(Nt(a)) is the times each action has been done
    # ucbF is the action selection function of Upper-Confidence-Bound Action Selection method,we want the Action At with max ucbF
    # initialize ucbF with 0,add and add small number 1e-5 to avoid the error when NumAction equal to 0
    # count ucbF for each action every step
    
    for j in range(ActionCount):
        ucbF[j] = Qta[j] + c * (math.sqrt((math.log(step+1))/( NumAction[j]+ 1e-5)))

    # choose the Action index whose ucbF is max
    action = ucbF.index(max(ucbF))

    temp=rewardReturn(step, action)
    NumAction[action] += 1
    reward[action] += temp
    Qta[action] = reward[action] / NumAction[action]
    
    return temp



############################## ALGORITHM PART #############################################


############################epsilon greedy result#################
#initialize the reward list and set up epsilon
#   traceReturn_eps:return of each step of epsilon greedy method
#   cumulativereturn_eps:cumulative return of each step of epsilon greedy method
initial()
epsilon = [0,0.1,0.5,0.8]
traceReturn_eps = np.zeros((len(epsilon),stepCount))
cumulativereturn_eps=np.zeros((len(epsilon),stepCount))

for eps in epsilon:
    for i in range(stepCount):
        r = eps_Greedy(eps, i)
        cumulative+=r
        traceReturn_eps[epsilon.index(eps),i]=r
        cumulativereturn_eps[epsilon.index(eps),i]=cumulative



for i in range(len(epsilon)):
        plt.plot(traceReturn_eps[i,:], label='$\epsilon = %.02f$' % (epsilon[i]))
plt.xlabel('steps')
plt.ylabel('reward')
plt.title("eps method")
plt.legend()



plt.figure(figsize=(30, 10))

for i in range(len(epsilon)):
        plt.plot(cumulativereturn_eps[i,:], label='$\epsilon = %.02f$' % (epsilon[i]))
plt.xlabel('steps')
plt.ylabel('cumulative reward')
plt.title("epsilon greedy method")
plt.legend()



############################greedy decay result###################
#initialize the reward list
#   traceReturn_decayEps:return of each step of greedy decay method
#   cumulativereturn_decayEps:cumulative return of each step of greedy decay method
initial()
traceReturn_decayEps =[]
cumulativereturn_decayEps=[]

for i in range(stepCount):
    r = decayEps_Greedy(i)
    cumulative+=r
    traceReturn_decayEps.append(r)
    cumulativereturn_decayEps.append(cumulative)



plt.plot(traceReturn_decayEps)
plt.xlabel('steps')
plt.ylabel('reward')
plt.title("greedy decay method")



plt.plot(cumulativereturn_decayEps)
plt.xlabel('steps')
plt.ylabel('cumulative reward')
plt.title("greedy decay method")



###########################UCB result##################
#initialize the reward list
#   traceReturn_UCB:return of each step of greedy decay method
#   cumulativereturn_UCB:cumulative return of each step of greedy decay method
# initialize ucbF with 0
initial()
ucbF = [0.0] * ActionCount
traceReturn_UCB= []
cumulativereturn_UCB=[]

for i in range(stepCount):
    r = UCB(i, c=2)
    cumulative+=r
    traceReturn_UCB.append(r)
    cumulativereturn_UCB.append(cumulative)



plt.plot(traceReturn_UCB)
plt.xlabel('steps')
plt.ylabel('reward')
plt.title("UCB method")




plt.plot(cumulativereturn_UCB)
plt.xlabel('steps')
plt.ylabel('cumulative reward')
plt.title("UCB method")







