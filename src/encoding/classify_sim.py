import dynamiqs as dq
import jax.numpy as jnp

import numpy as np
from src.helper_functions import rho2vec
from src.TK_basics import PSD_rho
from tqdm import tqdm
from qutip import Qobj, wigner
from multiprocess.pool import Pool
from sklearn.linear_model import RidgeClassifier, LogisticRegression
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
from bisect import bisect_left

chi = 0.177
kerr = 3e-3

T1 = 120
T2 = 20

Tphi = - 1 / (1 / 2 / T1 - 1 / T2)
cavT1 = 1.21e3
nbar_cav = 0.094
alpha = 114.97

chi_m = 1.48
Td = 2.84e-1
T1_m = 103
T2_m = 13
Tphi_m = - 1 / (1 / 2 / T1_m - 1 / T2_m)
A_parity = 6530617.972423978e-6

sigma_par, chop_par = 16e-3, 4
def linspace(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1))
def thermal_dq(N, n):
    i = jnp.arange(N)
    beta = jnp.log(1.0 / n + 1.0)
    diags = jnp.exp(-beta * i)
    diags = diags / jnp.sum(diags)
    rm = jnp.diag(diags)
    return rm

def pulse_parity(t):
    global sigma_par, chop_par
    sigma, chop = sigma_par, chop_par
    t0 = sigma*chop/2
    t1 = sigma*chop*3/2 + Td
    return jnp.heaviside(sigma*chop - t, 0) * jnp.exp(- 1 / 2 * (t - t0) ** 2 / sigma ** 2)\
        + jnp.heaviside(t - Td - sigma*chop, 0) * jnp.exp(- 1 / 2 * (t - t1) ** 2 / sigma ** 2)
class ClassifySim:
    def __init__(self, compression=16, t_total=17.5, pc_real=0, pc_imag=0.175, pq_real=0, pq_imag=0.175,
                 cdim=-1, dm=10, intervals=3):

        self.compression = compression
        self.__loadData()
        self.t_total = t_total

        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag


        self.qdim = 2
        self.dm = dm
        if cdim == -1:
            cdim = dm + 3 # padding dimensions for simulation
                          # if using 10 dimensions, evolving with 13 dimensions
                          # and truncating results in more accurate states
        self.cdim = cdim
        self.tsave = jnp.linspace(0, t_total, intervals+1)
        self.intervals = intervals

        self.pns_path = f'..\\..\\data\\disps\\pns_disps{self.dm},d=1.npy'

        self.E = self.__getE(cdim)

    def __loadData(self):
        self.trainX_total = jnp.array(np.load("..\\..\\data\\compressed16\\trainX16.npy"))
        self.testX_total = jnp.array(np.load("..\\..\\data\\compressed16\\testX16.npy"))
        self.trainy_total = jnp.array(np.load("..\\..\\data\\images\\train_y.npy"))
        self.testy_total = jnp.array(np.load("..\\..\\data\\images\\test_y.npy"))
    def getHt(self, op, val): # TODO: SHUFFLE
        values = []
        for i in range(self.trainSize):
            img = self.trainX[i]
            img = (img + 1) / 2
            img = val * img
            img = jnp.pad(img, (1, 1), 'constant')
            values.append(img)
        for i in range(self.testSize):
            img = self.testX[i]
            img = (img + 1) / 2
            img = val * img
            img = jnp.pad(img, (1, 1), 'constant')
            values.append(img)
        values = np.array(values)
        times = jnp.linspace(0, self.t_total, values.shape[1] + 1)
        return dq.pwc(times, values, op)



    def __getE(self, dim):
        E = []
        proj = dq.fock_dm(dim, self.dm - 1)
        displacements = np.array([dq.displace(dim, i) for i in np.load(self.pns_path)], dtype=complex)
        for disp in displacements:
            E.append(np.hstack(np.array(disp @ proj @ dq.dag(disp))))
        return np.array(E)
    def __getEp(self, dim):
        return np.linalg.pinv(self.__getE(dim))
    def evolve(self, trainSize, testSize=0, c_ops=False, shuffle=True):

        self.trainSize = trainSize
        self.testSize = testSize

        if type(trainSize) is list:
            self.trainSize = len(trainSize)
            self.items_train = np.array(trainSize)
        else:
            if shuffle:
                self.items_train = np.random.choice(len(self.trainX_total), self.trainSize)
            else:
                self.items_train = np.arange(self.trainSize)

        if type(testSize) is list:
            self.testSize = len(testSize)
            self.items_test = np.arange(self.testSize)
        else:
            if shuffle:
                self.items_test = np.random.choice(len(self.testX_total), self.testSize)
            else:
                self.items_test = np.arange(self.testSize)

        self.trainX = self.trainX_total[self.items_train]
        self.trainy = self.trainy_total[self.items_train]

        self.testX = self.testX_total[self.items_test]
        self.testy = self.testy_total[self.items_test]

        a = dq.tensor(dq.eye(self.qdim), dq.destroy(self.cdim))
        ad = dq.dag(a)
        q = dq.tensor(dq.destroy(self.qdim), dq.eye(self.cdim))
        qd = dq.dag(q)

        # chi -> chi_m, T1 -> T1_m, T2 -> T2_m
        H0 = -2 * jnp.pi * chi * ad @ a @ qd @ q - jnp.pi*kerr * ad @ ad @ a @ a
        H = H0 + self.getHt(2 * jnp.pi*(qd + q), self.pq_imag) + \
                 self.getHt(2j * jnp.pi * (q - qd), self.pq_real) + \
                 self.getHt(2 * jnp.pi * (ad + a), self.pc_imag) + \
                 self.getHt(2j * jnp.pi * (a - ad), self.pc_real)


        if c_ops:
            rho_i = dq.tensor(dq.fock_dm(self.qdim, 0), thermal_dq(self.cdim, nbar_cav))

            jops = [
                jnp.sqrt(1 / T1) * q,
                jnp.sqrt(2 / Tphi) * qd @ q,
                jnp.sqrt((1 + nbar_cav) / cavT1) * a,
                jnp.sqrt(nbar_cav / cavT1) * ad,
            ]
            result = dq.mesolve(H, jops, rho_i, self.tsave)
        else:
            rho_i = dq.tensor(dq.fock(self.qdim, 0), dq.fock(self.cdim, 0))
            print("Evolving states (simulation)...")
            result = dq.sesolve(H, rho_i, self.tsave)

        self.trainX_evolved = result.states[:self.trainSize, 1:]
        self.testX_evolved = result.states[self.trainSize:, 1:]
        self.trainX_states = dq.ptrace(result.states[:self.trainSize, 1:], 1, (self.qdim, self.cdim))
        self.testX_states = dq.ptrace(result.states[self.trainSize:, 1:], 1, (self.qdim, self.cdim))

        self.vectorize()
        return self


    def vectorize(self):
        self.trainX_vecs = self.E @ np.array(self.trainX_states.reshape(self.trainSize*self.intervals, self.cdim**2).T)
        self.trainX_vecs = np.real(self.trainX_vecs).T.reshape(self.trainSize, self.intervals*self.E.shape[0])

        self.testX_vecs = self.E @ np.array(self.testX_states.reshape(self.testSize*self.intervals, self.cdim**2).T)
        self.testX_vecs = np.real(self.testX_vecs).T.reshape(self.testSize, self.intervals*self.E.shape[0])

        return self

    def addError(self, n):
        if n < 1:
            sigma_trainX = n
            sigma_testX = n
        else:
            sigma_trainX = np.sqrt(np.abs(self.trainX_vecs * (1 - self.trainX_vecs))) / np.sqrt(n)
            sigma_testX = np.sqrt(np.abs(self.testX_vecs * (1 - self.testX_vecs))) / np.sqrt(n)


        err_trainX = np.random.normal(0, sigma_trainX)
        err_testX = np.random.normal(0, sigma_testX)

        self.trainX_vecz = self.trainX_vecs + err_trainX
        self.testX_vecz = self.testX_vecs + err_testX

        return self
    def __getrho(self, i):
        vecs = np.split(i, self.intervals)
        vecz = []
        rhoz = []
        for vec in vecs:
            rho_vec = self.Ep @ vec
            rho = rho_vec.reshape(self.Edim, self.Edim).T
            rho = rho / rho.trace()
            rho = PSD_rho(rho)
            rhoz.append(rho)
            vecz += list(rho2vec(PSD_rho(rho)))
        return vecz, rhoz

    def getRho(self, X):
        return list(map(self.__getrho, tqdm(X)))
    def mle(self, Edim=0):
        if not Edim:
            Edim = self.cdim
        self.Edim = Edim
        self.Ep = self.__getEp(Edim)
        print("Applying mle (simulation)...")
        if self.trainX_vecz.shape[0]:
            final_trainX_vecz, final_trainX_rhoz = zip(*self.getRho(self.trainX_vecz))
        else:
            final_trainX_vecz, final_trainX_rhoz = [], []

        if self.testX_vecz.shape[0]:
            final_testX_vecz, final_testX_rhoz = zip(*self.getRho(self.testX_vecz))
        else:
            final_testX_vecz, final_testX_rhoz = [], []

        self.trainX_final = np.array(final_trainX_vecz)
        self.testX_final = np.array(final_testX_vecz)

        self.trainX_rhos = np.array(final_trainX_rhoz)
        self.testX_rhos = np.array(final_testX_rhoz)
        return self

    def __addTrainX(self, X):
        return np.hstack((self.trainX, X))
    def __addTestX(self, X):
        return np.hstack((self.testX, X))
    def score(self, method='final' , init=True, rscore=False):
        clf = RidgeClassifier(alpha=0)
        score = -1
        if init:
            f = self.__addTrainX
            g = self.__addTestX
        else:
            f = g = lambda z: z
        if method == "final":
            clf.fit(f(self.trainX_final), self.trainy)
            score = clf.score(g(self.testX_final), self.testy)
        elif method in {"obs", "observables"}:
            clf.fit(f(self.trainX_vecz), self.trainy)
            score = clf.score(g(self.testX_vecz), self.testy)

        if rscore: # If return score set to true
            return score

        print(f"method={method}, init={init}:  {score}")
        return self

    def __getaxes(self, n):
        if n == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        elif n <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        elif n <= 9:
            fig, axes = plt.subplots(3, 3, figsize=(10,10))
        else:
            cols = 5
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = np.array(axes).reshape(-1)
        return fig, axes
    def wig(self, n=-1, scale=5, step=.5, seq=False):
        if n == -1:
            n = self.trainSize
        xvec = linspace(-scale, scale, step)
        yvec = linspace(-scale, scale, step)
        fig, axes = self.__getaxes(n)
        if seq:
            xvec /= np.sqrt(2)
            yvec /= np.sqrt(2)
            rho_i = dq.tensor(dq.fock_dm(self.qdim, 0), self.trainX_states)
            # rho_i = self.trainX_evolved
            disps = dq.tensor(dq.eye(self.qdim), jnp.array([[dq.displace(self.cdim, i + 1j * j) for i in xvec] for j in yvec]))
            disps_d = dq.dag(disps)
            states = jnp.einsum('ijkl,...lm,ijmn->...ijkn', disps_d, rho_i, disps, optimize=True) # for each rho, apply nxn displacements
            # Z = dq.tensor(dq.zero(self.qdim), dq.zero(self.cdim))
            a = dq.tensor(dq.eye(self.qdim), dq.destroy(self.cdim))
            ad = dq.dag(a)
            q = dq.tensor(dq.destroy(self.qdim), dq.eye(self.cdim))
            qd = dq.dag(q)
            H0 = -2 * jnp.pi * chi_m * ad @ a @ qd @ q - jnp.pi * kerr * ad @ ad @ a @ a
            H = H0 + dq.modulated(pulse_parity, jnp.pi * A_parity * 1j * (qd - q))
            jops = [
                jnp.sqrt(1 / T1_m) * q,
                jnp.sqrt(2 / Tphi_m) * qd @ q,
                jnp.sqrt((1 + nbar_cav) / cavT1) * a,
                jnp.sqrt(nbar_cav / cavT1) * ad,
            ]
            e_dm = dq.tensor(dq.fock_dm(self.qdim, 1), dq.eye(self.cdim))
            result = dq.mesolve(H, jops, states, jnp.array([0, Td + sigma_par * chop_par * 2]),
                                exp_ops=[e_dm])
            wigners = (2 * jnp.real(result.expects[:, :, :, :, 0, -1]) - 1)/jnp.pi
            for i in range(n):
                W = wigners[i][-1]
                axes[i].contourf(xvec*np.sqrt(2), yvec*np.sqrt(2), W, vmin=-1/np.pi, vmax=1/np.pi)
                axes[i].set_title(f"{self.trainy[i]}, (id: {self.items_train[i]})")
                axes[i].axis('off')
                #print(np.max(W))

        else:
            for i in range(n):
                W = wigner(dq.to_qutip(self.trainX_states[i][-1]), xvec, yvec)
                axes[i].contourf(xvec, yvec, W, vmin=-1/np.pi, vmax=1/np.pi)
                axes[i].set_title(f"{self.trainy[i]}, (id: {self.items_train[i]})")
                axes[i].axis('off')
                #print(np.max(W))
        fig.suptitle(f'seq: {seq}')
        plt.show()
        return self

    def __getTV(self, drive, index): # get times, values
        img = self.trainX[index]
        img = (img + 1) / 2
        img = drive * img
        img = np.pad(img, (1, 1), 'constant')
        times = np.linspace(0, self.t_total, len(img) + 1)
        return times, img

    def __smoothstep(self, var, index, grad, steps):
        drive = {"QubitI": self.pq_imag, "QubitQ": self.pq_real, "CavityI": self.pc_imag, "CavityQ": self.pc_real}[var]
        if not drive:
            return [0], [0]

        times, values = self.__getTV(drive, index)
        halfsteps = int(steps / 2)
        arrx, arry = [np.linspace(times[0], (times[0] + times[1]) / 2, halfsteps)], [np.zeros(halfsteps) + values[0]]
        grad *= drive
        for i in range(len(times) - 2):
            x = np.linspace((times[i] + times[i + 1]) / 2, (times[i + 1] + times[i + 2]) / 2, steps)
            dy = values[i + 1] - values[i]
            dx = abs(dy / grad)
            x_min = times[i + 1] - dx / 2
            mini = min(values[i], values[i + 1])
            maxi = max(values[i], values[i + 1])
            y = np.clip(values[i] + np.sign(dy) * grad * (x - x_min), mini, maxi)
            arrx.append(x)
            arry.append(y)
        arrx.append(np.linspace((times[-2] + times[-1]) / 2, times[-1], halfsteps))
        arry.append(np.zeros(halfsteps) + values[-1])
        return np.hstack(arrx), np.hstack(arry)

    def getDrives(self, index=0, grad=1, dt=1, invscale=1000, steps=1000):
        arr = []
        timescale = int(1000 / dt)
        timesteps = int(timescale * self.t_total + 1)
        for var in ["QubitI", "QubitQ", "CavityI", "CavityQ"]:
            x, y = self.__smoothstep(var, index, grad, steps)
            xnew = np.linspace(0, self.t_total, timesteps)
            ynew = np.array([y[bisect_left(x, t) - 1] for t in xnew])
            arr.append(ynew/invscale)

        return arr

