import os
import scipy as sp
import numpy as np
from tqdm import tqdm
from src.encoding.evolution import Evolve
from multiprocess.pool import Pool
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier, LogisticRegression
from qutip import wigner
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import tensorflow as tf
import tensorflow_datasets as tfds
from qutip import qeye, destroy, Qobj, displace, fock_dm
from src.TK_basics import PSD_rho

from src.helper_functions import rho2vec, vec2rho, factor_int
from matplotlib import pyplot as plt


class Classify:
    def __init__(self, compression=4,
                 t_total=2e-6, pc_real=0, pc_imag=0, pq_real=0, pq_imag=0,
                 cdim=-1, nsteps=2000, intervals=3, dataset="mnist", measurement='pns trace',
                 c_ops=True, anharm=True, kerr=True, ffilter=True, dm=10, method="2qubit"):
        self.compression = compression
        self.t_total = t_total
        self.nsteps = nsteps
        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag
        self.intervals = intervals
        self.dataset = dataset
        self.trained = False
        self.loaded = False
        self.measurement = measurement
        self.c_ops = c_ops
        self.anharm = anharm
        self.kerr = kerr
        self.ffilter = ffilter
        self.method = method
        self.dm = dm
        if cdim == -1:
            cdim = dm + 3
        self.cdim = cdim
        self.Ep = self.getEp()

        self.__loadData()
    def __loadData(self): # TODO: SAVE DIFFERENT DATASETS
        '''if self.dataset not in tfds.list_builders():
            print("No such dataset available.")
            return False

        ds_train, ds_test = tfds.load(self.dataset, split=['train', 'test'], as_supervised=True, shuffle_files=True)

        trainSize_total = ds_train.cardinality().numpy()
        testSize_total = ds_test.cardinality().numpy()
        ds_train = ds_train.shuffle(trainSize_total)
        ds_test = ds_test.shuffle(testSize_total)
        n = self.compression

        def reshape(image, label):
            return tf.reshape(image, [-1]), label

        ds_train = ds_train.map(reshape, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(reshape, num_parallel_calls=tf.data.AUTOTUNE)
        train_X, train_y, test_X, test_y = [], [], [], []
        print("Retrieving datasets...")
        for x, y in tqdm(ds_train.take(ds_train.cardinality())):
            train_X.append(x.numpy())
            train_y.append(y.numpy())
        for x, y in tqdm(ds_test.take(ds_test.cardinality())):
            test_X.append(x.numpy())
            test_y.append(y.numpy())'''
        train_X = np.load("..\\data\\images\\train_X.npy")
        test_X = np.load("..\\data\\images\\test_X.npy")
        train_y = np.load("..\\data\\images\\train_y.npy")
        test_y = np.load("..\\data\\images\\test_y.npy")

        pca = PCA(self.compression)
        train_X_pca = pca.fit_transform(train_X)
        test_X_pca = pca.transform(test_X)
        train_X = normalize(train_X_pca)
        test_X = normalize(test_X_pca)

        self.train_X_total = np.array(train_X)
        self.train_y_total = np.array(train_y)
        self.test_X_total = np.array(test_X)
        self.test_y_total = np.array(test_y)
        self.loaded = True
        return True

    def __getVec(self, img):
        e = Evolve(img, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag, self.cdim, self.nsteps,
                   self.intervals, self.measurement, self.c_ops, self.anharm, self.kerr, self.ffilter,
                   self.dm)
        return e.vector(self.method)

    def __getRho(self, img):
        e = Evolve(img, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag, self.cdim, self.nsteps,
                   self.intervals, self.measurement, self.c_ops, self.anharm, self.kerr, self.ffilter,
                   self.dm)
        return e.getRho_cav()
    def __multiprocess(self, train_imgs, test_imgs, max_pool=10):
        if max_pool <= 1:
            train_X_states = list(map(self.__getVec, tqdm(train_imgs)))
            test_X_states = list(map(self.__getVec, tqdm(test_imgs)))
        else:
            with Pool(max_pool) as p:
                train_X_states = list(
                    tqdm(
                        p.imap(self.__getVec,train_imgs),
                        total=self.trainSize

                    )
                )
                test_X_states = list(
                    tqdm(
                        p.imap(self.__getVec,test_imgs),
                        total=self.testSize

                    )
                )
        return train_X_states, test_X_states

    def train(self, trainSize, testSize, max_pool=10):
        self.trainSize = trainSize
        self.testSize = testSize
        if not self.loaded:   # Returns False if unable to loadData
            print("Data is not loaded")
            return self
        self.train_X = self.train_X_total[:trainSize]
        self.test_X = self.test_X_total[:testSize]
        self.train_y = self.train_y_total[:trainSize]
        self.test_y = self.test_y_total[:testSize]
        print("Evolving Hamiltonians for each Image...")
        train_X_states, test_X_states = self.__multiprocess(self.train_X, self.test_X, max_pool=max_pool)
        if self.method == "1qubit":
            train_X_g, train_X_e, train_X_p_g, train_X_p_e = zip(*train_X_states)
            test_X_g, test_X_e, test_X_p_g, test_X_p_e = zip(*test_X_states)


            self.train_X_g = np.array(train_X_g)
            self.train_X_e = np.array(train_X_e)
            self.train_X_p_g = np.array(train_X_p_g)
            self.train_X_p_e = np.array(train_X_p_e)

            self.test_X_g = np.array(test_X_g)
            self.test_X_e = np.array(test_X_e)
            self.test_X_p_g = np.array(test_X_p_g)
            self.test_X_p_e = np.array(test_X_p_e)
        elif self.method == "2qubit":
            self.train_X_processed = np.array(train_X_states)
            self.test_X_processed = np.array(test_X_states)
        self.trained = True
        return self


    def scoreL(self):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(self.train_X, self.train_y)
        return clf.score(self.test_X, self.test_y)

    def visualize(self, method, n, max_pool=10, wig_scale=5, shuffle=False, items=None):
        if items is None:
            if shuffle:
                items = np.random.choice(len(self.train_X_total), n)
            else:
                items = range(n)
        if n <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        elif n <= 9:
            fig, axes = plt.subplots(3, 3, figsize=(10,10))
        else:
            cols = 5
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = np.array(axes).reshape(-1)

        if method in {"wigner", "Wigner", "wig", "Wig"}:
            X_wig = self.train_X_total[items]
            if max_pool <= 1:
                X_rhos = list(map(self.__getRho, tqdm(X_wig)))
            else:
                with Pool(max_pool) as p:
                    X_rhos = list(
                        tqdm(
                            p.imap(self.__getRho, X_wig),
                            total=len(X_wig)

                        )
                    )
            xvec = np.linspace(-wig_scale, wig_scale, 100)
            yvec = np.linspace(-wig_scale, wig_scale, 100)
            for i in range(n):
                rho = X_rhos[i]
                W = wigner(rho, xvec, yvec)
                W = np.array(W)
                #W[W < 0.02] = 0
                axes[i].contourf(xvec, yvec, W)
                axes[i].set_title(f"{self.train_y_total[items[i]]}, ({items[i]})")
                axes[i].axis('off')

        elif method in {"input", "inp", "Input", "Inp", "compressed", "Compressed"}:
            for i in range(n):
                pixels = self.train_X_total[items[i]].reshape(*factor_int(self.compression))
                axes[i].imshow(pixels, cmap='gray')
                axes[i].set_title(f"{self.train_y_total[items[i]]}")
                axes[i].axis('off')
        plt.show()
        return self

    def getEp(self):
        if self.measurement in {"par", "parity", "par trace", "parity trace"}:
            displacements = np.load(f'..\\data\\disps\\disps2.npy')
            E = []
            a = destroy(self.cdim)
            P = (1j * np.pi * a.dag() * a).expm()
            for alpha in displacements:
                D = displace(self.cdim, alpha)
                E.append(np.hstack(D.dag() * P * D)[0])
        else:
            displacements = np.load(f'..\\data\\disps\\pns_disps{self.dm}.npy')
            E = []
            proj = fock_dm(self.cdim, self.dm - 1)
            for alpha in displacements:
                D = displace(self.cdim, alpha)
                E.append(np.hstack(D.dag() * proj * D)[0])
        return np.linalg.pinv(np.array(E))

    def getrho(self, i):
        split = self.intervals
        vecs_g = np.split(i[0], split)
        vecs_e = np.split(i[1], split)
        ps_g = np.split(i[2], split)
        ps_e = np.split(i[3], split)
        Ep = self.Ep
        cdim = self.cdim
        vec = []
        for vec_g, vec_e, pg, pe in zip(vecs_g, vecs_e, ps_g, ps_e):
            rho_g_vec = Ep @ vec_g
            rho_e_vec = Ep @ vec_e
            rho_g = np.hstack([c[..., None] for c in np.split(rho_g_vec, cdim)])
            rho_g = PSD_rho(Qobj(rho_g[:cdim, :cdim]).unit())
            rho_e = np.hstack([c[..., None] for c in np.split(rho_e_vec, cdim)])
            rho_e = PSD_rho(Qobj(rho_e[:cdim, :cdim]).unit())
            pg = pg / (pg + pe)
            pe = pe / (pg + pe)
            rho = (pg[0] * rho_g + pe[0] * rho_e).unit()
            vec += list(rho2vec(PSD_rho(rho)))
        return vec

    def getRho(self, X_g, X_e, p_g, p_e):
        max_pool = 13
        items = list(zip(X_g, X_e, p_g, p_e))
        with Pool(max_pool) as p:
            X_rho = list(
                tqdm(
                    p.imap(self.getrho, items),
                    total=len(items)

                )
            )
        return np.array(X_rho)

    def getrho2(self, i):
        vecs = np.split(i, self.intervals)
        vecz = []
        for vec in vecs:
            rho_vec = self.Ep @ vec
            rho = np.hstack([c[..., None] for c in np.split(rho_vec, self.cdim)])
            rho = Qobj(rho[:self.cdim, :self.cdim]).unit()

            vecz += list(rho2vec(PSD_rho(rho)))
        return vecz

    def getRho2(self, X):
        max_pool = 13
        items = X
        with Pool(max_pool) as p:
            X_rho = list(
                tqdm(
                    p.imap(self.getrho2, items),
                    total=len(items)

                )
            )
        return np.array(X_rho)

    def std(self, n, iterations=1):
        if self.method == "1qubit":
            trainX_g = self.train_X_g
            trainX_e = self.train_X_e
            trainX_pg = self.train_X_p_g
            trainX_pe = self.train_X_p_e
            testX_g = self.test_X_g
            testX_e = self.test_X_e
            testX_pg = self.test_X_p_g
            testX_pe = self.test_X_p_e
            trainy = self.train_y
            testy = self.test_y
            tx = self.train_X
            ttx = self.test_X
            results = []
            for i in range(iterations):
                if n < 1:
                    trainX_g_stds = n
                    trainX_e_stds = n
                    testX_g_stds = n
                    testX_e_stds = n
                else:
                    trainX_g_stds = np.sqrt(np.abs(trainX_g * (1 - trainX_g))) / np.sqrt(n)
                    trainX_e_stds = np.sqrt(np.abs(trainX_e * (1 - trainX_e))) / np.sqrt(n)
                    testX_g_stds = np.sqrt(np.abs(testX_g * (1 - testX_g))) / np.sqrt(n)
                    testX_e_stds = np.sqrt(np.abs(testX_e * (1 - testX_e))) / np.sqrt(n)

                trainX_g_noise = np.random.normal(0, trainX_g_stds)
                trainX_e_noise = np.random.normal(0, trainX_e_stds)
                testX_g_noise = np.random.normal(0, testX_g_stds)
                testX_e_noise = np.random.normal(0, testX_e_stds)

                trainX_g_est = trainX_g + trainX_g_noise
                trainX_e_est = trainX_e + trainX_e_noise
                testX_g_est = testX_g + testX_g_noise
                testX_e_est = testX_e + testX_e_noise
                trainX_pg_est = trainX_pg + trainX_pg
                trainX_pe_est = trainX_pe + trainX_pe
                testX_pg_est = testX_pg + testX_pg
                testX_pe_est = testX_pe + testX_pe

                trainX_est = self.getRho(trainX_g_est, trainX_e_est, trainX_pg_est, trainX_pe_est)
                testX_est = self.getRho(testX_g_est, testX_e_est, testX_pg_est, testX_pe_est)

                clf = RidgeClassifier(alpha=0)
                clf.fit(trainX_est, trainy)
                results.append(clf.score(testX_est, testy))
            results = np.array(results)
            if n < 1:
                print(f"std={n}: ", results.mean())
            else:
                print(f"samples={n}: ", results.mean())
            return self
        elif self.method == "2qubit":
            trainX = self.train_X_processed
            testX = self.test_X_processed
            trainy = self.train_y
            testy = self.test_y
            tx = self.train_X
            ttx = self.test_X
            results = []
            for i in range(iterations):
                if n < 1:
                    trainX_stds = n
                    testX_stds = n
                else:
                    trainX_stds = np.sqrt(np.abs(trainX * (1 - trainX))) / np.sqrt(n)
                    testX_stds = np.sqrt(np.abs(testX * (1 - testX))) / np.sqrt(n)
                trainX_noise = np.random.normal(0, trainX_stds)
                testX_noise = np.random.normal(0, testX_stds)

                trainX_est = trainX + trainX_noise
                testX_est = testX + testX_noise

                trainX_est = self.getRho2(trainX_est)
                testX_est = self.getRho2(testX_est)

                clf = RidgeClassifier(alpha=0)
                clf.fit(trainX_est, trainy)
                results.append(clf.score(testX_est, testy))
            results = np.array(results)
            if n < 1:
                print(f"std={n}: ", results.mean())
            else:
                print(f"samples={n}: ", results.mean())
            return self


