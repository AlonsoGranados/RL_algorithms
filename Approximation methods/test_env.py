import gym
import numpy as np
from gym.spaces import Discrete
from On_policy_MC_gradient import linear_online_RL

env = gym.make("MountainCar-v0")
# env = gym.make("CartPole-v1")
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
alg = linear_online_RL(state_space,action_space)
w = np.zeros((19,1))
# w = np.random.normal(size=(21,1))
G = 0

episodic_G = []
state = env.reset()
action = alg.epsilon_greedy(w,state)
best_G = -200
best_w = np.zeros((19,1))
Q = []


for episode in range(5000):
    if (episode % 100 == 0):
        print(episode)
    for l in range(1000):
        # if(episode > 9000):
        #     env.render()
        # action = env.action_space.sample() # your agent here (this takes random actions)

        # SARSA
        next_state, reward, done, info = env.step(action)
        Q.append(alg.linear_Q(w,state,action)[0][0])
        G += reward

        if done:
            if(best_G < G):
                print(episode)
                print(G)
                best_w = w
                best_G = G
            episodic_G.append(G)
            if(reward == 0):
                w = alg.semi_grad_step(w,state,action,None,None,reward, alpha=0.1/8,gamma=0.9,isTerminal=True)
            state = env.reset()
            action = alg.epsilon_greedy(w, state)
            G = 0
            break
        else:
            # continue
            next_action = alg.epsilon_greedy(w, next_state)
            w = alg.semi_grad_step(w, state, action, next_state,
                               next_action, reward, alpha=0.1/8, gamma=0.9, isTerminal=False)
            state = next_state
            action = next_action

print(np.median(episodic_G))
env.close()

import matplotlib.pyplot as plt
plt.plot(episodic_G)
plt.show()

plt.plot(Q)
plt.show()

print(w)
print(best_w)