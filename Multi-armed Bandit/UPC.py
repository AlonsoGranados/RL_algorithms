import numpy as np
import matplotlib.pyplot as plt

# Q_values: Estimated value of an action
# N_values: Number of actions chosen
# R_values: Rewards over time
# a: Selected action
# armedBandits: Distributions
# c: Degree of exploration

def UPC(steps,samples,number_bandits,c):

    total_R_values = np.zeros(steps)
    average_optimal_action = 0

    for s in range(samples):
        # True mean distributions
        armedBandits = np.random.randn(number_bandits)
        Best_action = np.argmax(armedBandits)
        Q_values = np.zeros(number_bandits)
        N_values = np.zeros(number_bandits)
        R_values = np.zeros(steps)
        optimal_action = np.zeros(steps)
        for i in range(steps):
            weighted_action = Q_values + c * np.sqrt(np.log(i+2)/N_values)
            a = np.argwhere(weighted_action==np.max(weighted_action)).flatten()
            index = np.random.randint(a.shape[0])
            a = a[index]

            if(a == Best_action):
                optimal_action[i] = 1
            R = np.random.randn(1) + armedBandits[a]
            R_values[i] = R
            N_values[a] += 1
            Q_values[a] += (R - Q_values[a])/N_values[a]
        # Average reward over time
        N = np.arange(steps)+1
        R_values = np.cumsum(R_values)/N
        total_R_values += R_values
        #Optimal action
        optimal_action = np.cumsum(optimal_action)/N
        average_optimal_action += optimal_action

    total_R_values /= samples
    average_optimal_action /= samples

    return total_R_values, average_optimal_action
