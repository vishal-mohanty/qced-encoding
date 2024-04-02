import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
img_len = 16
t_total = 2*1e-6
img = 0.238*1e6*np.array([-0.95048267, -0.61432123, -0.66091824, -0.97047204, -0.9688619, -0.6433533, -0.2585144, -0.8334389, -0.8335047, -0.29491967, -0.23424923, -0.7570116, -0.7973976, -0.67219114, -0.8983342, -0.9829087]
)
def pc(t):
    if t == t_total: return img[-1]
    return img[int(t / t_total * img_len)]
time = np.linspace(0, t_total, 1000)
signal = np.array([pc(t) for t in time])
W = fftfreq(signal.size, d=time[1] - time[0])
f_signal = rfft(signal)
cut_f_signal = f_signal.copy()
cut_f_signal[(W < 1e5)] = 0
cut_f_signal[(W > 11e6)] = 0
cut_signal = irfft(cut_f_signal)


plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.subplot(111)
plt.plot(time, signal)
plt.plot(time, cut_signal + signal.mean())
plt.show()