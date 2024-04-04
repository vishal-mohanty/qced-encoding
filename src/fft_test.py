import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
img_len = 16
t_total = 2*1e-6
img = 0.238*1e6*np.array([-0.97763366, -0.77427757, -0.6953913, -0.9231226, -0.8911458, -0.21218914, -0.027030647, -0.40206188, -0.5931872, -0.0155918, -0.2635038, -0.39135206, -0.77145445, -0.47355384, -0.7337942, -0.9259224]
)
def pc(t):
    if t == t_total: return img[-1]
    return img[int(t / t_total * img_len)]
time = np.linspace(0, t_total, 1000)
signal = np.array([pc(t) for t in time])
print(signal.mean())
W = fftfreq(signal.size, d=time[1] - time[0])
f_signal = rfft(signal)
cut_f_signal = f_signal.copy()
cut_f_signal[(W < 1e5)] = 0
cut_f_signal[(W > 20e6)] = 0
cut_signal = irfft(cut_f_signal)



plt.subplot(111)
plt.plot(time, signal)
plt.plot(time, cut_signal + signal.mean())
plt.show()