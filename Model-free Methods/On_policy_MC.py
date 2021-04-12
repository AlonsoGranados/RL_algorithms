import numpy as np
import gym
import matplotlib.pyplot as plt

def MC_on_policy():
    env = gym.make('Blackjack-v0')
    env.reset()
    Q = np.zeros((env.observation_space[0].n,env.observation_space[1].n,env.observation_space[2].n,env.action_space.n))
    returns = {}
    for i_episode in range(500000):
        episode_s = []
        episode_a = []
        episode_r = []
        first_time = {}
        observation = env.reset()
        episode_s.append(observation)
        # print(observation)
        for t in range(1000):
            action = env.action_space.sample()
            soft_e_policy(Q,observation)
            episode_a.append(action)
            # print(action)
            observation, reward, done, _ = env.step(action)
            episode_s.append(observation)
            episode_r.append(reward)
            # print(observation)
            # print(reward)
            if done:
                # print(reward)
                episode_r = np.array(episode_r)
                G = np.cumsum(episode_r[::-1])
                for t in range(len(episode_a)):
                    if(returns.__contains__('{0}_{1}_{2}_{3}'.format(episode_s[t][0],episode_s[t][1],episode_s[t][2],episode_a[t]))):
                        returns['{0}_{1}_{2}_{3}'.format(episode_s[t][0], episode_s[t][1], episode_s[t][2],
                                                         episode_a[t])].append(G[t])
                    else:
                        returns['{0}_{1}_{2}_{3}'.format(episode_s[t][0], episode_s[t][1], episode_s[t][2],
                                                         episode_a[t])] = [G[t]]
                    Q[episode_s[t][0], episode_s[t][1], episode_s[t][2]+0,episode_a[t]] = \
                        np.mean(returns['{0}_{1}_{2}_{3}'.format(episode_s[t][0], episode_s[t][1], episode_s[t][2],episode_a[t])])
                # print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()

    return Q

def soft_e_policy(Q,s):
    e = 0.1
    sample = np.random.uniform()
    if(sample > e):
        action = np.argwhere(Q[s[0], s[1], s[2]+0] == np.amax(Q[s[0], s[1], s[2]+0]))
        action = action.flatten()
        a = np.random.randint(action.shape[0])
        action = action[a]
    else:
        action = np.random.randint(0,1)
    return action

Q = MC_on_policy()

V = np.argmax(Q,axis=3)

print(V.shape)

x = np.arange(12, 21, 1)
y = np.arange(1, 12, 1)
y, x = np.meshgrid(y, x)
V = V[11:20,0:11,0]


print(x.shape)
print(y.shape)
print(V.shape)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, V, rstride=1, cstride=1, cmap=plt.cm.rainbow, linewidth=0, antialiased=False)
plt.show()

plt.imshow(V)
plt.show()