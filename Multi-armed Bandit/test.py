from e_greedy_algorithm import e_greedy_algorithm
from UPC import UPC
import matplotlib.pyplot as plt
steps = 2000
samples = 1000
number_bandits = 10

rewards = []
optimal_actions = []

# E_greedy_algorithm
average_reward, average_optimal_action = e_greedy_algorithm(steps, samples, number_bandits, 0.1)
rewards.append(average_reward)
optimal_actions.append(average_optimal_action)
# UPC
average_reward, average_optimal_action = UPC(steps, samples, number_bandits, 0.1)
rewards.append(average_reward)
optimal_actions.append(average_optimal_action)

# Plot Rewards
plt.title("Average reward over time")
plt.plot(rewards[0], label="E_greedy")
plt.plot(rewards[1], label="UPC")
plt.legend()
plt.show()

# Plot Optimal Actions
plt.title("Average optimal action")
plt.plot(optimal_actions[0], label="E_greedy")
plt.plot(optimal_actions[1], label="UPC")
plt.legend()
plt.show()