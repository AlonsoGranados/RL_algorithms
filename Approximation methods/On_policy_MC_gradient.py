import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class linear_online_RL:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.Q_values = np.zeros(self.action_space)
        # self.poly = PolynomialFeatures(2)
        self.tilts = self.generate_tilt()

    def generate_tilt(self):
        n = 8
        tilts = np.zeros(n*self.state_space)
        tilts[:n] = np.random.uniform(-1.2,0.6 - (1.8/8),size=n)
        tilts[n:] = np.random.uniform(-0.7, 0.7 - (1.4/8), size=n)
        return tilts

    def x(self,s,a):
        x = np.zeros(self.tilts.shape[0] + self.action_space)
        for i in range(int(self.tilts.shape[0]/2)):
            if(self.tilts[i] > s[0] and self.tilts[i] < s[0] + (1.8/8)):
                x[i] = 1
        for i in range(int(self.tilts.shape[0]/2)):
            i += int(self.tilts.shape[0]/2)
            if(self.tilts[i] > s[1] and self.tilts[i] < s[1] + (1.4/8)):
                x[i] = 1
        # x[:self.state_space, 0] = s
        # print(tilts[0])
        x[self.tilts.shape[0]+a] = 1
        # x = self.poly.fit_transform(np.reshape(x, (1, -1)))
        return x.reshape((-1,1))

    def linear_Q(self, w,s,a):
        return np.dot(w.T,self.x(s,a))

    # def gradient_step(w,ep_s,ep_a,ep_r,alpha):
    #     G = 0
    #     for t in range(len(ep_a)-1,-1,-1):
    #         G += ep_r[t]
    #         w += alpha*(G - linear_Q(w,ep_s[t],ep_a[t])) * x(ep_s[t],ep_a[t])
    #     return w

    def semi_grad_step(self,w,state,action,next_state,next_action,reward,alpha,gamma,isTerminal):
        if isTerminal:
            w += alpha * (reward - self.linear_Q(w,state,action)) * self.x(state,action)
        else:
            w += alpha * (reward + gamma * self.linear_Q(w,next_state,next_action)
                          - self.linear_Q(w,state,action)) * self.x(state,action)
        return w

    def epsilon_greedy(self,w,s):
        epsilon = np.random.uniform()
        if(epsilon < 0.1):
            return np.random.randint(self.action_space)
        else:
            for a in range(self.action_space):
                self.Q_values[a] = self.linear_Q(w, s, a)
            action = np.argwhere(self.Q_values == np.amax(self.Q_values))
            action = action.flatten()
            a = np.random.randint(action.shape[0])
            return action[a]