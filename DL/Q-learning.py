import numpy as np
import pandas as pd
import time
np.random.seed(2)
# 创建一个示例 DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police 90%选择最优动作，10%随机选择动作
ALPHA = 0.1     # learning rate 学习率
GAMMA = 0.9    # discount factor 折扣因子
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move,方便观察

def build_q_table(n_states,actions):
    table=pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions
    )
    #print(table)
    return table

#build_q_table(N_STATES,ACTIONS)

def choose_action(state,q_table):
    state_actions=q_table.iloc[state,:]
    if np.random.uniform()>EPSILON or state_actions.all()==0:
        action_name=np.random.choice(ACTIONS)
    else:
        action_name=state_actions.argmax()
    return action_name

def get_env_feedback(S,A):
    if A=='right':
        if S==N_STATES-2:
            S_='terminal'
            R=1
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R

def rl():
    q_table=build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        S=0
        is_terminated=False
        update_env(S,episode,step_counter)
        while not is_terminated:
            A=choose_action(S,q_table)
            S_,R=get_env_feedback(S,A)
            q_predict=q_table.loc[S,A]
            if S_!='terminal':
                q_target=R+GAMMA*q_table.iloc[S_,:].max()
            else:
                q_target=R
                is_terminated=True
            q_table.loc[S,A]+=ALPHA*(q_target-q_predict)
            S=S_
            update_env(S,episode,step_counter)
    return q_table

def update_env(S,episode,step_counter):
    
