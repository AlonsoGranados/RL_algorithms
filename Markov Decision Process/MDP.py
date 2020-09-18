import numpy as np
from gridworld import action_to_state

np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)

states_value = np.zeros((5,5))
gamma = 0.9
error = 1

# Dynamic Bellman Equation
while(error > 0.001):
    error = 0
    for j in range(states_value.shape[0]):
        for k in range(states_value.shape[1]):
            temp = states_value[j, k]
            states_value[j,k] = 0
            for a in range(4):
                r, x, y = action_to_state(a,j,k)
                if(x == j and y == k):
                    states_value[j, k] += 0.25 * (r + gamma * temp)
                else:
                    states_value[j,k] += 0.25 * (r + gamma * states_value[x,y])
            error = np.maximum(error,np.abs(temp-states_value[j,k]))

print(states_value)