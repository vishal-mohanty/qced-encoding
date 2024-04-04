import numpy as np
from qutip import tensor, destroy, qeye, fock, sigmax, sigmay, mesolve, Options, displace
import os
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
from xxhash import xxh3_128

chi = 1.359e6
Kerr = 6e3
T1 = 84e-6
T2 = 21e-6
Tphi = - 1 / (1 / 2 / T1 - 1 / T2)
cavT1 = 1.21e-3
nbar_cav = 0
nbar_qb = 0
Td = 284e-9

sigma, chop = 16e-9, 4

A = 6530617.972423978

u_e = fock(2, 1)

cur_path = os.path.dirname(__file__)
path = os.path.join(cur_path, '..\\..\\data\\disps.npy')
displacements = np.load(path)
vecs = os.path.join(cur_path, '..\\..\\data\\vecs\\')
def pulse(t, *args):
    global sigma, chop
    if t <= sigma*chop:
        t0 = sigma * chop / 2
        g = np.exp(- 1 / 2 * (t - t0) ** 2 / sigma ** 2)
        return g
    elif t <= Td + sigma*chop:
        return 0
    else:
        t0 = sigma * chop * 3/2 + Td
        g = np.exp(- 1 / 2 * (t - t0) ** 2 / sigma ** 2)
        return g
class Evolve:

    def __init__(self, img, t_total=2e-6, pc_real=0.238*1e6, pc_imag=0, pq_real=0.238*1e6, pq_imag=0,
                 N=11, nsteps=2000, intervals=5):
        self.img = np.array(img).flatten()
        self.t_total = t_total
        self.N = N
        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag
        self.nsteps = nsteps
        self.intervals = intervals
        self.img_len = len(self.img)
        self.nlist = np.linspace(0, self.t_total, nsteps)
        self.tlist = np.linspace(0, self.t_total, intervals+1)
        self.C = tensor(qeye(2), destroy(N))
        self.Q = tensor(destroy(2), qeye(N))
        self.Cd = self.C.dag()
        self.Qd = self.Q.dag()
        self.psi_i = tensor(fock(2, 0), fock(N, 0))
        self.rho_i = self.psi_i * self.psi_i.dag()
        self.hash = xxh3_128(str(self)).hexdigest()
        self.c_ops = [
            np.sqrt((1 + nbar_qb) / T1) * self.Q,       # Qubit Relaxation
            np.sqrt(nbar_qb / T1) * self.Qd,            # Qubit Thermal Excitations
            np.sqrt(2 / Tphi) * self.Qd * self.Q,       # Qubit Dephasing, changed
            np.sqrt((1 + nbar_cav) / cavT1) * self.C,   # Cavity Relaxation
            np.sqrt(nbar_cav / cavT1) * self.Cd,        # Cavity Thermal Excitations
        ]
    def __str__(self):
        return str([self.img, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag,
                    self.N, self.nsteps, self.intervals])

    def freq_filter(self, func):
        signal = np.array([func(t) for t in self.nlist])
        W = fftfreq(signal.size, d=self.nlist[1] - self.nlist[0])
        f_signal = rfft(signal)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(abs(W) > 12e6)] = 0
        cut_signal = irfft(cut_f_signal)

        def f(t, args):
            if t >= self.t_total: return cut_signal[-1]
            return cut_signal[int(t/self.t_total*cut_signal.size)]
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

    def vector(self):
        if os.path.isfile(vecfile := vecs + self.hash + ".npy"):
            return np.load(vecfile)
        options = Options(nsteps=self.nsteps)

        H_disp = -2*np.pi*chi * self.Cd * self.C * self.Qd * self.Q + 2*np.pi*Kerr/2 * self.Cd * self.Cd * self.C * self.C
        H = [H_disp,
             [2 *np.pi*(self.Q + self.Qd), self.freq_filter(self.__pq_real)],
             [2j*np.pi*(self.Q - self.Qd), self.freq_filter(self.__pq_imag)],
             [2 *np.pi*(self.C + self.Cd), self.freq_filter(self.__pc_real)],
             [2j*np.pi*(self.C - self.Cd), self.freq_filter(self.__pc_imag)]]
        states = mesolve(H, self.rho_i, tlist=self.tlist, c_ops=self.c_ops, options=options).states

        Hd = 2 * np.pi * A * 1j * (self.Qd - self.Q) / 2

        H_measure = [H_disp, [Hd, pulse]]
        vec = []
        for rho in states[1:]:
            for alpha in displacements:
                disp = tensor(qeye(2), displace(self.N, alpha))
                rho = disp.dag()*rho*disp
                rho = mesolve(H_measure, rho, tlist=[0, Td+sigma*chop*2], c_ops=self.c_ops).states[-1]

                rho_t = rho.ptrace(0)
                p_e = (rho_t * u_e * u_e.dag()).tr()
                par = (2 * p_e - 1).real
                vec.append(par)
        vec = np.array(vec)
        np.save(vecs+self.hash, vec)
        return vec


if __name__ == "__main__":
    img = np.array([-0.97763366, -0.77427757, -0.6953913, -0.9231226, -0.8911458, -0.21218914, -0.027030647,
                    -0.40206189, -0.5931872, -0.0155918, -0.2635038, -0.39135206, -0.77145445, -0.47355384,
                    -0.7337942, -0.9259224])
    print(Evolve(img).vector())