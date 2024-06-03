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
from qutip import qeye, destroy, tensor, displace
class Classify:
    def __init__(self, compression=4,
                 t_total=2e-6, pc_real=0, pc_imag=0, pq_real=0, pq_imag=0,
                 cdim=13, nsteps=2000, intervals=3, dataset="mnist", measurement='pns trace',
                 c_ops=True, anharm=True, ffilter=True, dm=10, method="2qubit"):
        self.compression = compression
        self.t_total = t_total
        self.nsteps = nsteps
        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag
        self.cdim = cdim
        self.intervals = intervals
        self.dataset = dataset
        self.trained = False

        self.measurement = measurement
        self.c_ops = c_ops
        self.anharm = anharm
        self.ffilter = ffilter
        self.method = method
        self.dm = dm
    def __loadData(self):
        if self.dataset not in tfds.list_builders():
            print("No such dataset available.")
            return False

        ds_train, ds_test = tfds.load(self.dataset, split=['train', 'test'], as_supervised=True, shuffle_files=True)

        trainSize_total = ds_train.cardinality().numpy()
        testSize_total = ds_test.cardinality().numpy()
        ds_train = ds_train.shuffle(trainSize_total)   # TODO: UNCOMMENT
        ds_test = ds_test.shuffle(testSize_total)      # TODO: UNCOMMENT
        if self.trainSize > trainSize_total:
            print(f"Maximum size of training data for dataset: '{self.dataset}' is {trainSize_total}")
            return False
        elif self.testSize > testSize_total:
            print(f"Maximum size of test data for dataset: '{self.dataset}' is {trainSize_total}")
            return False
        n = self.compression

        def reshape(image, label):
            return tf.reshape(image, [-1]), label

        ds_train = ds_train.map(reshape, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(reshape, num_parallel_calls=tf.data.AUTOTUNE)
        train_X, train_y, test_X, test_y = [], [], [], []
        print("Retrieving datasets...")
        for x, y in tqdm(ds_train.take(self.trainSize)):
            train_X.append(x.numpy())
            train_y.append(y.numpy())
        for x, y in tqdm(ds_test.take(self.testSize)):
            test_X.append(x.numpy())
            test_y.append(y.numpy())

        pca = PCA(n)
        train_X_pca = pca.fit_transform(train_X)
        test_X_pca = pca.transform(test_X)
        train_X = normalize(train_X_pca)
        test_X = normalize(test_X_pca)

        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)
        self.test_X = np.array(test_X)
        self.test_y = np.array(test_y)
        return True

    def __getVec(self, img):
        e = Evolve(img, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag, self.cdim, self.nsteps,
                   self.intervals, self.measurement, self.c_ops, self.anharm, self.ffilter,
                   self.dm)
        return e.vector(self.method)
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
        if not self.__loadData():   # Returns False if unable to loadData
            return self
        print("Evolving Hamiltonians for each Image...")
        train_X_states, test_X_states = self.__multiprocess(self.train_X, self.test_X, max_pool=max_pool)
        if self.method == "1qubit":
            train_X_g = []
            train_X_e = []
            test_X_g = []
            test_X_e = []
            train_X_p_g = []
            train_X_p_e = []
            test_X_p_g = []
            test_X_p_e = []
            for i in train_X_states:
                train_X_g.append(i[0])
                train_X_e.append(i[1])
                train_X_p_g.append(i[2])
                train_X_p_e.append(i[3])

            for i in test_X_states:
                test_X_g.append(i[0])
                test_X_e.append(i[1])
                test_X_p_g.append(i[2])
                test_X_p_e.append(i[3])


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
        return self


    def scoreR(self):
        mq = []
        clf = RidgeClassifier(alpha=0)
        clf.fit(self.train_X_processed, self.train_y)
        q_score = clf.score(self.test_X_processed, self.test_y)
        mq.append(q_score)
        return max(mq)
    def scoreL(self):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(self.train_X, self.train_y)
        return clf.score(self.test_X, self.test_y)
    def score(self, n_regressors=("Log",), q_regressors=("Ridge",), n_alpha=0.1):
        if not self.trained:
            print("Unable to score. Data has not been trained")
            return self
        reg_dict = {"LogCV": "Logistic Regression CV", "Log": "Logistic Regression", "Ridge": "Ridge Classifier"}
        for r in n_regressors:
            match r:
                case "Log":
                    clf = LogisticRegression(max_iter=1000)
                case "LogCV":
                    clf = LogisticRegressionCV(max_iter=1000)
                case "Ridge":
                    clf = RidgeClassifier(alpha=n_alpha)
                case _:
                    print("Regression type not valid")
                    continue
            clf.fit(self.train_X, self.train_y)
            score = clf.score(self.test_X, self.test_y)
            print(f"{reg_dict[r]} (Without Quantum processing) yields: {score}")

        for r in q_regressors:
            match r:
                case "Log":
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(self.train_X_processed, self.train_y)
                    q_score = clf.score(self.test_X_processed, self.test_y)
                case "LogCV":
                    clf = LogisticRegressionCV(max_iter=1000)
                    clf.fit(self.train_X_processed, self.train_y)
                    q_score = clf.score(self.test_X_processed, self.test_y)
                case "Ridge":
                    q_score = self.scoreR()
                case _:
                    print("Regression type not valid")
                    continue

            print(f"{reg_dict[r]} (With Quantum processing) yields: {q_score}")
        return self


