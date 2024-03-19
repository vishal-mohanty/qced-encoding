import numpy as np
from qutip import tensor, destroy, qeye, fock, sigmax, sigmay, mesolve, Options
import os

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
        self.tlist = np.linspace(0, self.t_total, intervals+1)
        self.C = tensor(destroy(N), qeye(2))
        self.Q = tensor(qeye(N), destroy(2))

        self.psi_i = tensor(fock(N, 0), fock(2, 0))
        self.rho_i = self.psi_i * self.psi_i.dag()

    def __pc_real(self, t, args):
        modified_img = self.img*self.pc_real
        if t > self.t_total: return modified_img[-1]
        return modified_img[int(np.floor(t / self.t_total * self.img_len))]
    def __pc_imag(self, t, args):
        modified_img = self.img*self.pc_imag
        if t > self.t_total: return np.conj(modified_img[-1])
        return np.conj(modified_img[int(np.floor(t / self.t_total * self.img_len))])
    def __pq_real(self, t, args):
        modified_image = self.img*self.pq_real
        if t > self.t_total: return modified_image[-1]
        return modified_image[int(np.floor(t / self.t_total * self.img_len))]
    def __pq_imag(self, t, args):
        modified_image = self.img*self.pq_imag
        if t > self.t_total: return np.conj(modified_image[-1])
        return np.conj(modified_image[int(np.floor(t / self.t_total * self.img_len))])

    def states(self):
        options = Options(nsteps=self.nsteps)

        H0 = -2*np.pi*self.g * self.C.dag() * self.C * self.Q.dag() * self.Q
        H = [H0, [2*np.pi*(self.Q + self.Q.dag()), self.__pq_real], [2*np.pi*1j*(self.Q - self.Q.dag()), self.__pq_imag],
             [2*np.pi*(self.C + self.C.dag()), self.__pc_real], [2*np.pi*1j*(self.C - self.C.dag()), self.__pc_imag]]
        rho = [p.ptrace([0]) for p in mesolve(H, self.rho_i, tlist=self.tlist, options=options).states]
        return rho[1:]