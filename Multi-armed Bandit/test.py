from e_greedy_algorithm import e_greedy_algorithm
from UPC import UPC
from gradient_bandit_algorithm import gradient_bandit_algorithm
import matplotlib.pyplot as plt

steps = 2000
samples = 1000
number_bandits = 10

rewards = []
optimal_actions = []

algorithms = ["E_greedy", "UPC", "Gradient"]

# E Greedy Algorithm
average_reward, average_optimal_action = e_greedy_algorithm(steps, samples, number_bandits, 0.1)
rewards.append(average_reward)
optimal_actions.append(average_optimal_action)

# UPC
average_reward, average_optimal_action = UPC(steps, samples, number_bandits, 2)
rewards.append(average_reward)
optimal_actions.append(average_optimal_action)

# Gradient Bandit Algorithm
average_reward, average_optimal_action = gradient_bandit_algorithm(steps, samples, number_bandits, 0.1)
rewards.append(average_reward)
optimal_actions.append(average_optimal_action)

# Plot Rewards
plt.figure(1)
plt.title("Average reward over time")
# Plot Optimal Actions
plt.figure(2)
plt.title("Average optimal action")
for i in range(len(algorithms)):
    plt.figure(1)
    plt.plot(rewards[i], label=algorithms[i])
    plt.figure(2)
    plt.plot(optimal_actions[i], label=algorithms[i])

plt.legend()
plt.figure(1)
plt.legend()
plt.show()
