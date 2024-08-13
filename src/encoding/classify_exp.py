import dynamiqs as dq

import numpy as np
from src.helper_functions import rho2vec
from src.TK_basics import PSD_rho
from tqdm import tqdm
from qutip import Qobj, wigner, fidelity
from multiprocess.pool import Pool
from sklearn.linear_model import RidgeClassifier, LogisticRegression
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
from bisect import bisect_left
import h5py
from src.encoding.classify_sim import ClassifySim

class ClassifyExp:
    def __init__(self, compression=16, t_total=17.5, pc_real=0, pc_imag=0.175, pq_real=0, pq_imag=0.175,
                 cdim=13, dm=10, intervals=1):
        self.compression=compression
        self.t_total=t_total

        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag

        self.cdim = cdim
        self.dm = dm
        self.intervals = intervals

        self.pns_path = f'..\\..\\data\\disps\\pns_disps{self.dm},d=1.npy'
        self.__loadData()

    def __loadData(self):
        self.trainX_total = np.load("..\\..\\data\\compressed16\\trainX16.npy")
        self.testX_total = np.load("..\\..\\data\\compressed16\\testX16.npy")
        self.trainy_total = np.load("..\\..\\data\\images\\train_y.npy")
        self.testy_total = np.load("..\\..\\data\\images\\test_y.npy")
    def setData(self, trainSize, testSize=0, shuffle=False):
        if type(trainSize) is list:
            trainRange = trainSize
            self.trainSize = len(trainSize)
        else:
            self.trainSize = trainSize
            if shuffle:
                trainRange = np.random.choice(len(self.trainX_total), self.trainSize)
            else:
                trainRange = np.arange(self.trainSize)
        if type(testSize) is list:
            testRange = testSize
            self.testSize = len(testSize)
        else:
            self.testSize = testSize
            if shuffle:
                testRange = np.random.choice(len(self.testX_total), self.testSize)
            else:
                testRange = np.arange(self.testSize)

        self.trainX = self.trainX_total[trainRange]
        self.trainy = self.trainy_total[trainRange]
        self.testX = self.testX_total[testRange]
        self.testy = self.testy_total[testRange]

        self.items_train = trainRange
        self.items_test = testRange
        return self

    def __getTV(self, drive, img): # get times, values
        img = (img + 1) / 2
        img = drive * img
        img = np.pad(img, (1, 1), 'constant')
        times = np.linspace(0, self.t_total, len(img) + 1)
        return times, img

    def __smoothstep(self, var, img, grad, steps):
        drive = {"QubitI": self.pq_imag, "QubitQ": self.pq_real, "CavityI": self.pc_imag, "CavityQ": self.pc_real}[var]
        if not drive:
            return [0], [0]

        times, values = self.__getTV(drive, img)
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

    def __getDrive(self, img, grad=1, dt=1, invscale=1000, steps=1000):
        arr = []
        timescale = int(1000 / dt)
        timesteps = int(timescale * self.t_total + 1)
        for var in ["QubitI", "QubitQ", "CavityI", "CavityQ"]:
            x, y = self.__smoothstep(var, img, grad, steps)
            xnew = np.linspace(0, self.t_total, timesteps)
            ynew = np.array([y[bisect_left(x, t) - 1] for t in xnew])
            arr.append(ynew/invscale)

        return arr
    def getDrives(self):
        print("Getting Drives...")
        train_drive, test_drive = [], []
        for i in tqdm(self.trainX):
            train_drive.append(self.__getDrive(i))
        for i in tqdm(self.testX):
            test_drive.append(self.__getDrive(i))
        train_drive = np.array(train_drive)
        test_drive = np.array(test_drive)
        return train_drive, test_drive
    def setObs(self, trainX_obs=0, testX_obs=0):
        # TODO: WRITE CODE TO SET OBS
        if type(trainX_obs) is int and trainX_obs == 0:
            self.trainX_obs = np.random.random(size=(self.trainSize, self.intervals*(self.dm**2 - 1)))
        else:
            self.trainX_obs = trainX_obs
        if type(testX_obs) is int and testX_obs == 0:
            self.testX_obs = np.random.random(size=(self.testSize, self.intervals*(self.dm**2 - 1)))
        else:
            self.testX_obs = testX_obs

        return self

    def __getE(self, dim):
        E = []
        proj = dq.fock_dm(dim, self.dm - 1)
        displacements = np.array([dq.displace(dim, i) for i in np.load(self.pns_path)], dtype=complex)
        for disp in displacements:
            E.append(np.hstack(np.array(disp @ proj @ dq.dag(disp))))
        return np.array(E)

    def __getEp(self, dim):
        return np.linalg.pinv(self.__getE(dim))

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
        print("Appltying mle (experiment)...")
        if self.trainX_obs.shape[0]:
            final_trainX_vecz, final_trainX_rhoz = zip(*self.getRho(self.trainX_obs))
        else:
            final_trainX_vecz, final_trainX_rhoz = [], []

        if self.testX_obs.shape[0]:
            final_testX_vecz, final_testX_rhoz = zip(*self.getRho(self.testX_obs))
        else:
            final_testX_vecz, final_testX_rhoz = [], []

        self.trainX_final = np.array(final_trainX_vecz)
        self.testX_final = np.array(final_testX_vecz)

        self.trainX_rhos = np.array(final_trainX_rhoz)
        self.testX_rhos = np.array(final_testX_rhoz)
        return self

    def __cmpfidelity(self, arr1, arr2):
        arr3 = np.zeros(arr1.shape[:2])
        for i in range(arr1.shape[0]):
            for j in range(arr1.shape[1]):
                arr3[i, j] = fidelity(Qobj(arr1[i, j]), Qobj(arr2[i, j]))
        return arr3

    def getFidelity(self):
        c = ClassifySim(self.compression, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag,
                        self.cdim, self.dm, self.intervals).evolve(list(self.items_train), list(self.items_test),
                        c_ops=True).addError(0).mle()
        self.trainX_cmp = self.__cmpfidelity(self.trainX_rhos, c.trainX_rhos)
        self.testX_cmp = self.__cmpfidelity(self.testX_rhos, c.testX_rhos)

        if self.trainSize:
            print("--Training Set--")
            print(f"Mean Fidelity: {np.mean(self.trainX_cmp)}\n")
        if self.testSize:
            print("--Test Set--")
            print(f"Mean Fidelity: {np.mean(self.trainX_cmp)}")

        if self.trainSize and self.testSize:
            return self.trainX_cmp, self.testX_cmp
        if self.trainSize:
            return self.trainX_cmp
        if self.testSize:
            return self.testX_cmp


    def __addTrainX(self, X):
        return np.hstack((self.trainX, X))

    def __addTestX(self, X):
        return np.hstack((self.testX, X))

    def score(self, method='final', init=False, rscore=False):  # TODO: HSTACK ORIGINAL
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
            clf.fit(f(self.trainX_obs), self.trainy)
            score = clf.score(g(self.testX_obs), self.testy)

        if rscore:  # If return score set to true
            return score

        print(f"method={method}, init={init}:  {score}")
        return self
