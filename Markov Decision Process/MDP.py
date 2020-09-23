import numpy as np


# DP Bellman Equation
def DP_bellman_equation(states_value,gamma,dynamics,p,policy):
    delta = 1
    while(delta > 0.01):
        delta = 0
        for i,value_s in enumerate(states_value):
                temp = states_value[i]
                states_value[i] = 0
                actions, next_states, rewards = dynamics(i)
                for k, a in enumerate(actions):
                    if(next_states[k] == i):
                        states_value[i] += policy(a,temp)*p(next_states[k], rewards[k], a, i) * (
                                rewards[k] + gamma * temp)
                    else:
                        states_value[i] += policy(a,temp)*p(next_states[k], rewards[k], a, i) * (
                                    rewards[k] + gamma * states_value[next_states[k]])
                delta = np.maximum(delta,np.abs(temp-states_value[i]))
    return states_value