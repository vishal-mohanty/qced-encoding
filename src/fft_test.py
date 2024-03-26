import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
img_len = 16
t_total = 2
img = np.array([-0.9783678, -0.6780691, -0.5631608, -0.955075, -0.8387757, -0.29129118, -0.4497577, -0.7874726, -0.766538, -0.067662895, -0.14904398, -0.6202192, -0.954375, -0.6898675, -0.7004742, -0.93246067]
)
def pc(t):
    if t == t_total: return img-1
    return img[int(np.floor(t / t_total * img_len))]
time = np.linspace(0,2,1000)
signal = np.array([pc(t) for t in time])


W = fftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)
cut_f_signal = f_signal.copy()
cut_f_signal[(W<0.01)] = 0
cut_f_signal[(W>30)] = 0
cut_signal = irfft(cut_f_signal)

plt.subplot(211)
plt.plot(time,signal)
plt.subplot(212)
plt.plot(time,cut_signal)
plt.show()