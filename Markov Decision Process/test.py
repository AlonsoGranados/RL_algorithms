from gridworld import p
from gridworld import dynamics
from gridworld import random_policy
import numpy as np
from MDP import DP_bellman_equation
from Value_Iteration import Value_iteration

np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)

# GridWorld
# States value for random policy
state_values = np.zeros((5*5))
gamma = 0.9
final_values = DP_bellman_equation(state_values,gamma,dynamics,p,random_policy)
print(final_values)

# Value Iteration
state_values = np.zeros((5*5))
gamma = 0.9
final_values = Value_iteration(state_values,gamma,dynamics,p)
print(final_values)

