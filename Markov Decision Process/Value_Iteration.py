import numpy as np

def Value_iteration(states_value, gamma, dynamics, p):
    delta = 1
    while(delta > .1):
        delta = 0
        for i, v_state in enumerate(states_value):
            actions, next_states, rewards = dynamics(i)
            max = 0
            for k, a in enumerate(actions):
                temp = p(next_states[k],rewards[k],a,i) * (rewards[k] + gamma * states_value[next_states[k]])
                max = np.maximum(temp,max)
            # print(max)
            delta = np.maximum(delta, np.abs(states_value[i]-max))
            states_value[i] = max
    return states_value
