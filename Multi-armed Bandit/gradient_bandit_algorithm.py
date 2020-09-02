import numpy as np

def gradient_bandit_algorithm(steps,samples,number_bandits, alpha):
    total_R_values = np.zeros(steps)
    average_optimal_action = 0

    for s in range(samples):
        #True mean distributions
        armedBandits = np.random.randn(number_bandits)
        Best_action = np.argmax(armedBandits)
        average_reward = 0
        H_values = np.zeros(number_bandits)
        R_values = np.zeros(steps)
        prob = np.ones(number_bandits)/number_bandits
        optimal_action = np.zeros(steps)
        for i in range(steps):
            a = np.random.choice(number_bandits, 1, p=prob)
            # print(a)
            if(a == Best_action):
                optimal_action[i] = 1
            R = np.random.randn(1) + armedBandits[a]
            average_reward += (R - average_reward)/(i+1)
            temp_H = H_values[a]
            H_values += - alpha*(R - average_reward)* prob
            H_values[a] = temp_H + alpha*(R - average_reward)*(1-prob[a])
            R_values[i] = R
            prob = np.exp(H_values)/np.sum(np.exp(H_values))
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
