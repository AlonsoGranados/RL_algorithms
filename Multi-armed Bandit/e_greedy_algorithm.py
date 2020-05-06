import numpy as np
import matplotlib.pyplot as plt

#Parameters for Multi-armed Bandits
steps = 1000
samples = 2000
number_bandits = 10
e = 0.1

total_R_values = np.zeros(steps)
total_armedBandits = np.zeros(number_bandits)
total_Q_values = np.zeros(number_bandits)

for s in range(samples):
    armedBandits = np.random.randn(number_bandits)
    Q_values = np.zeros(number_bandits)
    N_values = np.zeros(number_bandits)
    R_values = np.zeros(steps)
    for i in range(steps):
        random_action = np.random.rand(1)
        if(random_action > e):
            a = np.argwhere(Q_values==np.max(Q_values)).flatten()
            index = np.random.randint(a.shape[0])
            a = a[index]
        else:
            a = np.random.randint(10)
        R = np.random.randn(1) + armedBandits[a]
        R_values[i] = R
        N_values[a] += 1
        Q_values[a] += (R - Q_values[a])/N_values[a]
    N = np.arange(steps)+1
    R_values = np.cumsum(R_values)/N
    total_R_values += R_values
    total_Q_values += Q_values
    total_armedBandits += armedBandits

total_R_values /= samples
total_armedBandits /= samples
total_Q_values /= samples

plt.plot(R_values)
plt.show()

print(total_armedBandits)
print(total_Q_values)
