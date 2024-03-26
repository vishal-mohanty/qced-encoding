import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
img_len = 16
t_total = 2
img = []
def pc(t):
    return img[int(np.floor(t / t_total * img_len))]
time = np.linspace(0,10,2000)
signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time)

W = fftfreq(signal.size, d=time[1]-time[0])
f_signal = rfft(signal)

# If our original signal time was in seconds, this is now in Hz
cut_f_signal = f_signal.copy()
cut_f_signal[(W<6)] = 0

cut_signal = irfft(cut_f_signal)