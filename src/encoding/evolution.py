import numpy as np
from qutip import tensor, destroy, qeye, fock, sigmax, sigmay, mesolve, Options
import os
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
class Evolve:
    cur_path = os.path.dirname(__file__)
    def __init__(self, img, t_total, pc_real, pc_imag, pq_real, pq_imag, N, g, nsteps, intervals):
        self.img = np.array(img).flatten()
        self.t_total = t_total
        self.N = N
        self.g = g
        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag
        self.nsteps = nsteps
        self.img_len = len(self.img)
        self.nlist = np.linspace(0, self.t_total, nsteps)
        self.tlist = np.linspace(0, self.t_total, intervals+1)
        self.C = tensor(destroy(N), qeye(2))
        self.Q = tensor(qeye(N), destroy(2))

        self.psi_i = tensor(fock(N, 0), fock(2, 0))
        self.rho_i = self.psi_i * self.psi_i.dag()

    def freq_filter(self, func):
        signal = np.array([func(t) for t in self.nlist])
        W = fftfreq(signal.size, d=self.nlist[1] - self.nlist[0])
        f_signal = rfft(signal)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W < 1e5)] = 0
        cut_f_signal[(W > 11e6)] = 0
        cut_signal = irfft(cut_f_signal)

        def f(t, args):
            if t >= self.t_total: return cut_signal[-1] + signal.mean()
            return cut_signal[int(t/self.t_total*cut_signal.size)] + signal.mean()
        return f

    def __pc_real(self, t):
        modified_img = self.img*self.pc_real
        if t >= self.t_total: return modified_img[-1]
        return modified_img[int(t / self.t_total * self.img_len)]

    def __pc_imag(self, t):
        modified_img = self.img*self.pc_imag
        if t >= self.t_total: return np.conj(modified_img[-1])
        return modified_img[int(t / self.t_total * self.img_len)]

    def __pq_real(self, t):
        modified_image = self.img*self.pq_real
        if t >= self.t_total: return modified_image[-1]
        return modified_image[int(t / self.t_total * self.img_len)]

    def __pq_imag(self, t):
        modified_image = self.img*self.pq_imag
        if t >= self.t_total: return np.conj(modified_image[-1])
        return modified_image[int(np.floor(t / self.t_total * self.img_len))]

    def states(self):
        options = Options(nsteps=self.nsteps)

        H0 = -2*np.pi*self.g * self.C.dag() * self.C * self.Q.dag() * self.Q
        H = [H0,
             [2 *np.pi*(self.Q + self.Q.dag()), self.freq_filter(self.__pq_real)],
             [2j*np.pi*(self.Q - self.Q.dag()), self.freq_filter(self.__pq_imag)],
             [2 *np.pi*(self.C + self.C.dag()), self.freq_filter(self.__pc_real)],
             [2j*np.pi*(self.C - self.C.dag()), self.freq_filter(self.__pc_imag)]]
        rho = [p.ptrace([0]) for p in mesolve(H, self.rho_i, tlist=self.tlist, options=options).states]
        return rho[1:]
