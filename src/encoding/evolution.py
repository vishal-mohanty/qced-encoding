import numpy as np
from qutip import tensor, destroy, qeye, fock, fock_dm, mesolve, Options, displace, thermal_dm
import os
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
from xxhash import xxh3_128

chi = 0.177e6
#chi=1.3e6
#chi = 0
Kerr = 3e3
T1 = 120e-6
T2 = 20e-6
Tphi = - 1 / (1 / 2 / T1 - 1 / T2)
cavT1 = 1.21e-3
nbar_cav = 0.094
nbar_qb = 0
Td = 2.84e-7
w = 114.97e6

T1_m = 103e-6
T2_m = 13e-6
chi_m = 1.48e6
w_m = 175.62e6

sigma_parity, chop_parity = 16e-9, 4
sigma_pns, chop_pns = 250e-9, 4
A_parity = 6530617.972423978
A_pns = 417959.5502351346



mean = np.array([-0.97037244, -0.79086132, -0.70268262, -0.91902397, -0.89255285, -0.45840487,
                 -0.33903784, -0.80714807, -0.86354743, -0.41002755, -0.34796831, -0.83375346,
                 -0.9248745,  -0.643027,   -0.67912024, -0.94334359])

cur_path = os.path.dirname(__file__)
parity_path = os.path.join(cur_path, '..\\..\\data\\disps\\disps2.npy')

#vecs = os.path.join(cur_path, '..\\..\\data\\vecs4\\')


def pulse_pns(t, *arg):
    global sigma_pns, chop_pns
    sigma, chop = sigma_pns, chop_pns
    t0 = sigma * chop / 2
    g = np.exp(- 1 / 2 * (t - t0) ** 2 / sigma ** 2)

    return g
def pulse_parity(t, *args):
    global sigma_parity, chop_parity
    sigma, chop = sigma_parity, chop_parity
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
    i = 0
    def __init__(self, img, t_total=2e-6, pc_real=0.0, pc_imag=0.0, pq_real=0.0, pq_imag=0.0,
                 cdim=13, nsteps=5000, intervals=3, measurement='pns trace',
                 c_ops=True, anharm=True, ffilter=True,
                 dm=10):
        self.anharm = anharm
        if anharm:
            self.qdim = 3
        else:
            self.qdim = 2
        self.img = (img+1)/2
        #self.img = img
        self.t_total = t_total
        self.cdim = cdim
        self.dm = dm
        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag
        self.nsteps = nsteps
        self.intervals = intervals
        self.img_len = len(self.img)
        self.nlist = np.linspace(0, self.t_total, nsteps)
        self.tlist = np.linspace(0, self.t_total, intervals+1)
        self.C = tensor(qeye(self.qdim), destroy(cdim))
        self.Q = tensor(destroy(self.qdim), qeye(cdim))
        self.Cd = self.C.dag()
        self.Qd = self.Q.dag()
        self.rho_i = tensor(fock_dm(self.qdim, 0), thermal_dm(cdim, nbar_cav))
        self.hash = xxh3_128(str(self)).hexdigest()
        self.c_ops = [
            np.sqrt((1 + nbar_qb) / T1) * self.Q,       # Qubit Relaxation
            np.sqrt(nbar_qb / T1) * self.Qd,            # Qubit Thermal Excitations
            np.sqrt(2 / Tphi) * self.Qd * self.Q,       # Qubit Dephasing, changed
            np.sqrt((1 + nbar_cav) / cavT1) * self.C,   # Cavity Relaxation
            np.sqrt(nbar_cav / cavT1) * self.Cd,        # Cavity Thermal Excitations
        ] if c_ops else []
        self.measurement = measurement
        pns_path = os.path.join(cur_path, f'..\\..\\data\\disps\\pns_disps{self.dm}.npy')
        match measurement:
            case 'parity' | 'parity trace' | 'par' | 'par trace':
                self.displacements = np.load(parity_path)
            case 'photon number' | 'photon_number' | 'pns' | \
                 'pns trace' | 'photon number trace' | 'photon_number trace':
                self.displacements = np.load(pns_path)
            case _:
                self.displacements = []
        self.g = fock_dm(self.qdim, 0)
        self.e = fock_dm(self.qdim, 1)
        self.gt = tensor(self.g, qeye(cdim))
        self.et = tensor(self.e, qeye(cdim))
        self.ffilter = ffilter
        self.rho = None
    def __str__(self):
        return str([self.img, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag,
                    self.cdim, self.nsteps, self.intervals])

    def freq_filter(self, func):
        signal = np.array([func(t) for t in self.nlist])
        signal = np.pad(signal, (1, 1), 'constant')
        W = fftfreq(signal.size, d=self.nlist[1] - self.nlist[0])
        f_signal = rfft(signal)
        cut_f_signal = f_signal.copy()
        if self.ffilter: cut_f_signal[(abs(W) > 30e6)] = 0
        cut_signal = irfft(cut_f_signal)
        cut_signal = cut_signal #- cut_signal[0]
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



    def vector(self, method="2qubit"):
        '''if os.path.isfile(vecfile := vecs + self.hash + ".npy"):
            return np.load(vecfile)'''
        options = Options(nsteps=self.nsteps)

        H_disp = -2*np.pi*chi * self.Cd * self.C * self.Qd * self.Q + \
                   2*np.pi*Kerr/2 * self.Cd * self.Cd * self.C * self.C

        if self.anharm: H_disp += -2*np.pi*w/2 * self.Qd*self.Qd*self.Q*self.Q
        H = [H_disp,
             [2 *np.pi*(self.Q + self.Qd), self.freq_filter(self.__pq_imag)],
             [2j*np.pi*(self.Q - self.Qd), self.freq_filter(self.__pq_real)],
             [2 *np.pi*(self.C + self.Cd), self.freq_filter(self.__pc_imag)],
             [2j*np.pi*(self.C - self.Cd), self.freq_filter(self.__pc_real)]]
        states = mesolve(H, self.rho_i, tlist=self.tlist, c_ops=self.c_ops, options=options).states

        # Parity Protocol
        H_disp_m = -2*np.pi*chi_m * self.Cd * self.C * self.Qd * self.Q + \
                   2*np.pi*Kerr/2 * self.Cd * self.Cd * self.C * self.C
        if self.anharm: H_disp_m += -2 * np.pi * w_m / 2 * self.Qd * self.Qd * self.Q * self.Q
        Hd_parity = 2 * np.pi * A_parity * 1j * (self.Qd - self.Q) / 2
        H_parity = [H_disp_m, [Hd_parity, pulse_parity]]

        # Photon number protocol TODO: TRY DIFFERENT Probabilities (i.e. P0, P1, P2, ...)
        Qo = tensor(destroy(self.qdim), qeye(self.cdim), qeye(self.qdim))
        Qn = tensor(qeye(self.qdim), qeye(self.cdim), destroy(self.qdim))
        Qod = Qo.dag()
        Qnd = Qn.dag()

        Cn = tensor(qeye(self.qdim), destroy(self.cdim), qeye(self.qdim))
        Cnd = Cn.dag()

        H_disp_m2 = -2*np.pi*chi_m * Cnd * Cn * Qnd * Qn + \
                   2*np.pi*Kerr/2 * Cnd * Cnd * Cn * Cn
        H_disp2 = -2*np.pi*chi * Cnd * Cn * Qod * Qo + \
                   2*np.pi*Kerr/2 * Cnd * Cnd * Cn * Cn
        H02 = H_disp_m2 + H_disp2 - 2*np.pi*-chi_m*(self.dm-1)*Qnd*Qn
        Hd_pns2 = 2*np.pi*A_pns*1j*(Qnd - Qn)
        H_pns2 = [H02, [Hd_pns2, pulse_pns]]
        et2 = tensor(qeye(self.qdim), qeye(self.cdim), self.e)
        e_ops_e2 = [et2]

        '''H0 = H_disp_m - 2 * np.pi * -chi_m*(self.dm-1) * self.Qd * self.Q
        Hd_pns = 2 * np.pi * A_pns * 1j * (self.Qd - self.Q)
        H_pns = [H0, [Hd_pns, pulse_pns]]'''

        e_ops_e = [self.et]
        e_ops_g = [self.gt]

        # Parity Direct Trace
        a = destroy(self.cdim)
        P = (1j*np.pi*a.dag()*a).expm()
        vec_g = []
        vec_e = []
        vec = []
        p_g = []
        p_e = []
        proj = fock_dm(self.cdim, self.dm-1)
        self.rhos = states[1:]
        if method == "1qubit":
            for rho in states[1:]:
                if self.measurement == "dm":
                    rho = rho.ptrace(1).full()
                    for i, j in zip(*np.triu_indices(self.dm)):
                        if i == j:
                            #print(i, rho[i, j].real)
                            pass
                        if i == j == self.dm - 1:
                            continue
                        vec.append(rho[i, j].real)
                        if i != j:
                            vec.append(rho[i, j].imag)
                else:
                    rho_g = (self.gt*rho).ptrace(1).unit()
                    rho_e = (self.et*rho).ptrace(1).unit()
                    rho_q = rho.ptrace(0)
                    p_g.append((self.g * rho_q).tr().real)
                    p_e.append((self.e * rho_q).tr().real)

                    for alpha in self.displacements:
                        disp = tensor(qeye(self.qdim), displace(self.cdim, alpha))
                        #rho_d = disp * tensor(self.g, rho.ptrace(1)).unit() * disp.dag()
                        rho_g_d = disp * tensor(self.g, rho_g) * disp.dag()
                        rho_e_d = disp * tensor(self.e, rho_e) * disp.dag() # TODO: REWRITE FOR H_PNS2
                        match self.measurement:
                            case 'parity' | 'par':
                                po_g = mesolve(H_parity, rho_g_d, tlist=[0, Td + sigma_parity * chop_parity * 2],
                                              c_ops=self.c_ops, e_ops=e_ops_e, options=Options(nsteps=3000)).expect[0][-1]
                                po_e = mesolve(H_parity, rho_e_d, tlist=[0, sigma_pns * chop_pns],
                                              c_ops=self.c_ops, e_ops=e_ops_g, options=Options(nsteps=3000)).expect[0][-1]

                                o_g = 2 * po_g - 1
                                o_e = 2 * po_e - 1
                            case 'parity trace' | 'par trace':
                                o_g = (P * rho_g_d.ptrace(1)).tr()
                                o_e = (P * rho_e_d.ptrace(1)).tr()
                            case 'photon number' | 'photon_number' | 'pns':
                                o_g = mesolve(H_pns2, rho_g_d, tlist=[0, sigma_pns*chop_pns],
                                            c_ops=self.c_ops, e_ops=e_ops_e, options=Options(nsteps=3000)).expect[0][-1]
                                o_e = mesolve(H_pns2, rho_e_d, tlist=[0, sigma_pns * chop_pns],
                                              c_ops=self.c_ops, e_ops=e_ops_g, options=Options(nsteps=3000)).expect[0][-1]

                            case 'pns trace' | 'photon number trace' | 'photon_number trace':
                                o_g = (proj*rho_g_d.ptrace(1)).tr()
                                o_e = (proj*rho_e_d.ptrace(1)).tr()
                                #p = (proj*rho_d.ptrace(1)).tr()
                            case _:
                                o_g = 0
                                o_e = 0
                        vec_g.append(o_g.real)
                        vec_e.append(o_e.real)
                        #vec_p.append(p.real)
            vec = [vec_g, vec_e, p_g, p_e]
            #vec = np.array(vec)
            #np.save(vecs+self.hash, vec)
            return vec
        elif method == "2qubit":
            for rho in states[1:]:
                if self.measurement == "dm":
                    rho = rho.ptrace(1).full()
                    for i, j in zip(*np.triu_indices(self.dm)):
                        if i == j:
                            pass
                        if i == j == self.dm - 1:
                            continue
                        vec.append(rho[i, j].real)
                        if i != j:
                            vec.append(rho[i, j].imag)
                else:
                    rho_m = tensor(rho, self.g)
                    rho_t = tensor(self.g, rho.ptrace(1)).unit()
                    for alpha in self.displacements:
                        match self.measurement:
                            case 'parity' | 'par':
                                disp_m = tensor(qeye(self.qdim), displace(self.cdim, alpha), qeye(self.qdim))
                                rho_d_m = disp_m * rho_m * disp_m.dag()
                                p_e = mesolve(H_parity, rho_d_m, tlist=[0, Td + sigma_parity * chop_parity * 2],
                                              e_ops=e_ops_e).expect[0][-1]
                                p = 2 * p_e - 1
                            case 'parity trace' | 'par trace':
                                disp_t = tensor(qeye(self.qdim), displace(self.cdim, alpha))
                                rho_d_t = disp_t * rho_t * disp_t.dag()
                                p = (P * rho_d_t.ptrace(1)).tr()
                            case 'photon number' | 'photon_number' | 'pns':
                                disp_m = tensor(qeye(self.qdim), displace(self.cdim, alpha), qeye(self.qdim))
                                rho_d_m = disp_m * rho_m * disp_m.dag()
                                p = mesolve(H_pns2, rho_d_m, tlist=[0, sigma_pns * chop_pns],
                                            e_ops=e_ops_e2, options=Options(nsteps=5000)).expect[0][-1]

                            case 'pns trace' | 'photon number trace' | 'photon_number trace':
                                disp_t = tensor(qeye(self.qdim), displace(self.cdim, alpha))
                                rho_d_t = disp_t * rho_t * disp_t.dag()
                                p = (proj * rho_d_t.ptrace(1)).tr()
                            case _:
                                p = 0
                        vec.append(p.real)
            vec = np.array(vec)
            # np.save(vecs+self.hash, vec)
            return vec
    def rho_test(self):
        return self.rhos

# TODO: COMPARE PHOTON NUMBER TRACE TO PHOTON NUMBER MESOLVE
# TODO: CREATE SEPARATE CASES FOR PHOTON NUMBER TRACE AND PARITY TRACE
if __name__ == "__main__":
    img = np.array([0.19031461653056986, -0.5433694189241283, 0.941050527248425, 0.4814215388610822, 0.5052590530838083,
                    -0.4881473132004496, 0.18178470184430385, -0.35487379853721646, 0.5686678766949724,
                    0.36226119031810744, 0.1553727142224298, -0.6545945224962656, -0.2925330621674876, 0.1809035742188693,
                    0.59519396673181, -0.1256754626669716])
    x = 1e6 * 0.5
    e = Evolve(img, measurement='pns trace', cdim=13, dm=10, t_total=2e-6, anharm=True, c_ops=True,
                      ffilter=True, intervals=1, nsteps=10000, pc_imag=x, pq_imag=x)
    e.vector('2qubit')
    rho = e.rho_test()
    rho = rho.ptrace(1)
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in rho]))