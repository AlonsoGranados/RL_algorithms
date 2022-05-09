import numpy as np
import gym
import matplotlib.pyplot as plt

def MC_on_policy():
    env = gym.make('Blackjack-v0')

    Q = np.zeros((env.observation_space[0].n,env.observation_space[1].n,env.observation_space[2].n,env.action_space.n))

    returns = {}
    for i_episode in range(100000):
        S = []
        A = []
        R = []

        observation = env.reset()
        S.append(observation)

        for t in range(1000):
            action = soft_e_policy(Q,observation)
            observation, reward, done, _ = env.step(action)

            # Store into trajectory
            A.append(action)
            S.append(observation)
            R.append(reward)

            if done:
                R = np.array(R)
                G = np.cumsum(R[::-1])
                for t in range(len(A)):
                    if(returns.__contains__('{0}_{1}_{2}_{3}'.format(S[t][0],S[t][1],S[t][2],A[t]))):
                        returns['{0}_{1}_{2}_{3}'.format(S[t][0], S[t][1], S[t][2],
                                                         A[t])].append(G[t])
                    else:
                        returns['{0}_{1}_{2}_{3}'.format(S[t][0], S[t][1], S[t][2],
                                                         A[t])] = [G[t]]

                    Q[S[t][0], S[t][1], int(S[t][2]),A[t]] = \
                        np.mean(returns['{0}_{1}_{2}_{3}'.format(S[t][0], S[t][1], S[t][2],A[t])])
                break
    env.close()

    return Q

def soft_e_policy(Q,s):
    e = 0.1
    sample = np.random.uniform()
    if(sample > e):
        action = np.argwhere(Q[s[0], s[1], int(s[2])] == np.amax(Q[s[0], s[1], int(s[2])]))
        action = action.flatten()
        a = np.random.randint(action.shape[0])
        action = action[a]
    else:
        action = np.random.randint(2)
    return action

Q = MC_on_policy()

print(Q.shape)

V = np.max(Q,axis=3)
print(V)

print(V.shape)

x = np.arange(12, 21, 1)
y = np.arange(2, 12, 1)
y, x = np.meshgrid(y, x)
V = V[12:21,1:11,0]


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