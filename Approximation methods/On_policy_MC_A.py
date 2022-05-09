import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def x(S,action,env):
    A = np.zeros(env.action_space.n)
    A[action] = 1
    feature_vector = np.zeros(S.shape[0] + A.shape[0])
    feature_vector[:S.shape[0]] = S
    feature_vector[S.shape[0]:] = A
    poly = PolynomialFeatures(2)
    feature_vector = poly.fit_transform(np.reshape(feature_vector,(1,-1)))
    # print(feature_vector.shape)
    feature_vector = np.reshape(feature_vector,-1)
    return feature_vector

def Q(S,action,w,env):
    feature_vector = x(S,action,env)
    return np.dot(w,feature_vector)

def soft_e_policy(S,w,env,e):
    sample = np.random.uniform()

    Q_values = np.zeros(env.action_space.n)
    for i in range(env.action_space.n):
        Q_values[i] = Q(S,i,w,env)

    if(sample > e):
        action = np.argwhere(Q_values == np.amax(Q_values))
        action = action.flatten()
        a = np.random.randint(action.shape[0])
        action = action[a]
        # print(Q_values)
        # print(action)
    else:
        action = np.random.randint(env.action_space.n)
    return action

def Sarsa_on_policy():
    total_g = 0
    alpha = 0.1
    gamma = 0.1
    e = 0.1
    env = gym.make('MountainCar-v0')
    w = np.zeros(21)
    for episode in range(1000):
        G = 0
        if(episode%100 == 0):
            print(G)
            print(episode)

        S = env.reset()
        action = soft_e_policy(S, w, env,e)
        for t in range(1000):
            # print(action)
            if(episode > 200):
                env.render()
            next_S, R, done, _ = env.step(action)
            # print(action)
            G += R
            # print(observation)
            # print(reward)
            if done:
                # print(G)
                total_g += G
                # print(reward)

                feature_vector = x(S, action, env)

                w = w + alpha * (R - Q(S,action,w,env) ) * feature_vector
                break
            else:

                feature_vector = x(S,action,env)
                next_action = soft_e_policy(next_S, w, env,e)

                w = w + alpha * (R + gamma * Q(next_S, next_action, w,env) - Q(S, action, w,env)) * feature_vector

                action = next_action
                S = next_S

    env.close()
    print(total_g)
    return w

def test(w):
    env = gym.make("MountainCar-v0")
    observation = env.reset()
    G = 0
    for episode in range(10):
        for _ in range(10000):

            env.render()
            action = soft_e_policy(observation, w, env,0)  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)
            G += reward

            if done:
                print(G)
                G = 0
                observation = env.reset()
                break

w = Sarsa_on_policy()

# w = np.array([-4.94518794e+02, -3.15189836e+04, 1.78936849e+03, -5.43017752e+03,
#               2.71793279e+03, -7.60205282e+01, -4.18498266e+02, -2.48195282e+05,
#               -1.58280881e+05, 2.09222065e+04, 9.24667185e+04,  1.81935274e+04,
#               -4.97125110e+04, -7.61947808e+03,  4.66322911e+03, -1.01854650e+04,
#               1.45638821e+03, 3.32980279e+02, 8.98783219e+03, 2.62935213e+04,
#               4.41703036e+02, -5.87188056e+03, -8.87868937e+03, -5.30445380e+02,
#               3.24837817e+03, -7.60205282e+01, 0.00000000e+00, -4.18498266e+02])


test(w)


print(w)

