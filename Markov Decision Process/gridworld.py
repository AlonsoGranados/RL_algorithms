import numpy as np

grid = np.zeros((5,5))
prob = np.zeros([50,4,4,50])

def random_policy(a,s):
    return 1/4

def next_states(s):
    if(s == 3):
        return 3

def reward_index(r):
    if(r == 0):
        return 0
    elif(r == -1):
        return 1
    elif(r == 5):
        return 2
    else:
        return 3

def p(s,r,a,v):
    return prob[s,reward_index(r),a,v]

def dynamics(current_state):
    actions = np.arange(4)
    next_states = np.zeros(4,dtype=int)
    rewards = np.zeros(4)
    for i,a in enumerate(actions):
        rewards[i],next_states[i] = action_to_state(a,current_state)
        prob[next_states[i],reward_index(rewards[i]),a,current_state] = 1
    return actions,next_states,rewards

def action_to_state(a,s):
    if (s == 1):
        return 10, 21
    if (s == 3):
        return 5, 13
    if(a == 0 and s % 5 == 4):
        return -1, s
    elif(a == 1 and  s < 5):
        return -1, s
    elif (a == 2 and s % 5 == 0):
        return -1, s
    elif (a == 3 and s > 19):
        return -1, s
    if(a == 0):
        s += 1
    elif (a == 1):
        s -= 5
    elif (a == 2):
        s -= 1
    elif (a == 3):
        s += 5
    return 0, s




