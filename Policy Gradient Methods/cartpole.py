import gym
import numpy as np
from REINFORCE import sample_policy
from REINFORCE import simple_REINFORCE

env = gym.make("CartPole-v1")

# Initialization
state = env.reset()
ep_s = [state]
ep_a = []
ep_r = []
theta = np.zeros((6, 1))
episodic_G = []
G = 0
best_theta = theta
best_G = 0

for _ in range(100000):

    env.render()
    # REINFORCE
    action = sample_policy(best_theta, state, 2)
    # Uniform policy
    # action = env.action_space.sample() # your agent here (this takes random actions)

    state, reward, done, info = env.step(action)

    ep_a.append(action)
    ep_s.append(state)
    ep_r.append(reward)
    G += reward
    if done:
        if (best_G < G):
            best_theta = theta
            best_G = G
        episodic_G.append(G)

        theta = simple_REINFORCE(theta, ep_s, ep_a, ep_r, alpha=0.0001)

        # Restart Episode
        G = 0
        state = env.reset()
        ep_s = [state]
        ep_a = []
        ep_r = []
env.close()

print(np.mean(episodic_G))

import matplotlib.pyplot as plt

k = 100
episodic_G = np.array(episodic_G)
a = episodic_G[0:-k + 1]
for i in range(1, k - 1, 1):
    a = a + episodic_G[i:-k + i + 1]
a = a / k
plt.plot(a)
plt.show()
print(best_theta)
print(best_G)
