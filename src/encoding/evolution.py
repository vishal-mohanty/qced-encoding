import numpy as np
from qutip import tensor, destroy, qeye, fock, sigmax, sigmay, mesolve, Options
import os

class Evolve:
    cur_path = os.path.dirname(__file__)
    def __init__(self, img, t_total, coef_pc, coef_pq, N, g, nsteps, intervals):
        self.img = np.array(img).flatten()
        self.t_total = t_total
        self.N = N
        self.g = g
        self.coef_pc = coef_pc
        self.coef_pq = coef_pq
        self.nsteps = nsteps
        self.img_len = len(self.img)
        self.tlist = np.linspace(0, self.t_total, intervals+1)
        self.a = tensor(destroy(N), qeye(2))
        self.e = tensor(qeye(N), fock(2, 1))
        self.sx = tensor(qeye(N), sigmax())
        self.sy = tensor(qeye(N), sigmay())

        self.psi_i = tensor(fock(N, 0), fock(2, 0))
        self.rho_i = self.psi_i * self.psi_i.dag()

    def __pc(self, t, args):
        modified_img = self.img*self.coef_pc
        if t > self.t_total: return modified_img[-1]
        return modified_img[int(np.floor(t / self.t_total * self.img_len))]
    def __pcc(self, t, args):
        modified_img = self.img*self.coef_pc
        if t > self.t_total: return np.conj(modified_img[-1])
        return np.conj(modified_img[int(np.floor(t / self.t_total * self.img_len))])
    def __pq(self, t, args):
        modified_image = self.img*self.coef_pq
        if t > self.t_total: return modified_image[-1]
        return modified_image[int(np.floor(t / self.t_total * self.img_len))]
    def __pqc(self, t, args):
        modified_image = self.img*self.coef_pq
        if t > self.t_total: return np.conj(modified_image[-1])
        return np.conj(modified_image[int(np.floor(t / self.t_total * self.img_len))])

    def states(self):
        H0 = -self.g / 2 * self.a.dag() * self.a * self.e * self.e.dag()
        options = Options(nsteps=self.nsteps)
        H = [H0, [self.a.dag(), self.__pc], [self.a, self.__pcc], [self.sx, self.__pq], [self.sy, self.__pqc]]
        rho = [p.ptrace([0]) for p in mesolve(H, self.rho_i, tlist=self.tlist, options=options).states]
        return rho[1:]