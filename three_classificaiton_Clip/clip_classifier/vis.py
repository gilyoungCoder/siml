import matplotlib.pyplot as plt
import numpy as np

# Data
time_steps = np.array([5, 15, 25, 35, 45])
acc_random = np.array([10, 10, 10, 10, 10])
acc_trained = np.array([18, 18, 14, 14, 10])
acc_direct = np.array([92, 94, 94, 90, 84])

# Simple smoothing with moving average (window size 3)
def smooth(y):
    kernel = np.ones(3) / 3
    return np.convolve(y, kernel, mode='same')

smooth_random = smooth(acc_random)
smooth_trained = smooth(acc_trained)
smooth_direct = smooth(acc_direct)

# Plot
plt.figure()
plt.plot(time_steps, smooth_random, marker='o', label='Random Init Projection')
plt.plot(time_steps, smooth_trained, marker='o', label='Trained Projection')
plt.plot(time_steps, smooth_direct, marker='o', label='Direct CLIP')
plt.xlabel('Time Step (t)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Diffusion Time Step')
plt.legend()
plt.grid(True)
plt.show()
