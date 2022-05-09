import numpy as np

# linear parameters
def x(s,a):
    x = np.zeros((6,1))
    x[:4,0] = s
    x[4+a,0] = 1
    return x

def linear_softmax(theta,s,action_space):
    policy = np.zeros(action_space)
    for a in range(action_space):
        policy[a] = np.dot(theta.T, x(s,a))
    policy = policy - np.max(policy)
    policy = np.exp(policy)
    policy = policy / np.sum(policy)
    return policy

def sample_policy(theta,s,action_space):
    policy = linear_softmax(theta,s,action_space)
    action = np.random.choice(2,1,p=policy)
    return action[0]


def grad_linear_softmax(s,a,theta,action_space):
    policy = linear_softmax(theta,s,action_space)
    E = 0
    for b in range(action_space):
        E += policy[b] * x(s,b)
    grad_J = x(s,a) - E
    return grad_J

def simple_REINFORCE(theta,ep_s,ep_a,ep_r,alpha):
    action_space = 2
    G = 0
    for t in range(len(ep_a)-1,-1,-1):
        G += ep_r[t]
        theta += alpha * G * grad_linear_softmax(ep_s[t],ep_a[t],theta,action_space)
    return theta

def baseline_REINFORCE(theta,ep_s,ep_a,ep_r,alpha):
    action_space = 2
    G = 0
    for t in range(len(ep_a)-1,-1,-1):
        G += ep_r[t]
        theta += alpha * G * grad_linear_softmax(ep_s[t],ep_a[t],theta,action_space)
    return theta