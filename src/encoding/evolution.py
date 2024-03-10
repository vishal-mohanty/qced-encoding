import numpy as np
from qutip import tensor, destroy, qeye, fock, sigmax, sigmay, mesolve, wigner, qload, qsave, displace, qfunc
import os
from xxhash import xxh128

class Evolve:
    cur_path = os.path.dirname(__file__)
    def __init__(self, img, t_total, t_steps, coef_pc, coef_pq, N, g):
        self.img = np.array(img).flatten()
        self.t_total = t_total
        self.N = N
        self.g = g
        self.coef_pc = coef_pc
        self.coef_pq = coef_pq
        self.t_steps = t_steps
        h = xxh128(str([self.img, t_total, t_steps, coef_pc, coef_pq, N, g])).hexdigest()
        self.save_path = os.path.relpath(
            f'data\\quantum_states\\{h}'
            , self.cur_path)
        self.img_len = len(self.img)
        self.img = np.append(self.img, [0, 0])

        self.a = tensor(destroy(N), qeye(2))
        self.e = tensor(qeye(N), fock(2, 1))
        self.sx = tensor(qeye(N), sigmax())
        self.sy = tensor(qeye(N), sigmay())

        self.psi_i = tensor(fock(N, 0), fock(2, 0))
        self.rho_i = self.psi_i * self.psi_i.dag()

    def __pc(self, t):
        modified_img = self.img*self.coef_pc
        if t > self.t_total: return modified_img[-1]
        return modified_img[int(np.floor(t / self.t_total * self.img_len))]

    def __pq(self, t):
        modified_image = self.img*self.coef_pq
        if t > self.t_total: return modified_image[-1]
        return modified_image[int(np.floor(t / self.t_total * self.img_len))]

    def __H(self, t, args):
        H0 = -self.g / 2 * self.a.dag() * self.a * self.e * self.e.dag()
        p_c = self.__pc(t)
        p_q = self.__pq(t)
        return H0 + p_c * self.a.dag() + np.conj(p_c) * self.a + p_q * self.sx + np.conj(p_q) * self.sy

    def states(self):
        if os.path.isfile(self.save_path + ".qu"):
            return qload(self.save_path)
        tlist = np.linspace(0, self.t_total, self.t_steps)
        rho = [p.ptrace([0]) for p in mesolve(self.__H, self.rho_i, tlist).states]
        qsave(rho, os.path.abspath(self.save_path))

        return rho