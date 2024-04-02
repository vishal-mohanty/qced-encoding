import os
import numpy as np
from tqdm import tqdm
from src.encoding.evolution import Evolve
from multiprocess.pool import Pool
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier, LogisticRegression
from qutip import wigner
import tensorflow as tf
import tensorflow_datasets as tfds
from qutip import qeye, destroy, tensor, displace
class Classify:
    cur_path = os.path.dirname(__file__)
    def __init__(self, compression=4,
                 t_total=12, pc_real=0.238*1e6, pc_imag=0, pq_real=0.238*1e6, pq_imag=0, N=11, g=1.4*1e6, nsteps=1000, intervals=1, dataset="mnist"):
        self.compression = compression
        self.t_total = t_total
        self.nsteps = nsteps
        self.pc_real = pc_real
        self.pc_imag = pc_imag
        self.pq_real = pq_real
        self.pq_imag = pq_imag
        self.N = N
        self.g = g
        self.intervals = intervals
        self.dataset = dataset
        self.trained = False
        self.processed = False

        path = os.path.join(self.cur_path, 'disps.npz')
        disps = np.load(path)['arr_0']
        self.displacements = [displace(self.N, d) for d in disps]

        a = destroy(self.N)
        self.P = (1j * np.pi * a.dag() * a).expm()



    def __loadData(self):
        if self.dataset not in tfds.list_builders():
            print("No such dataset available.")
            return False

        ds_train, ds_test = tfds.load(self.dataset, split=['train', 'test'], as_supervised=True, shuffle_files=True)

        trainSize_total = ds_train.cardinality().numpy()
        testSize_total = ds_test.cardinality().numpy()
        ds_train = ds_train.shuffle(trainSize_total)
        ds_test = ds_test.shuffle(testSize_total)
        if self.trainSize > trainSize_total:
            print(f"Maximum size of training data for dataset: '{self.dataset}' is {trainSize_total}")
            return False
        elif self.testSize > testSize_total:
            print(f"Maximum size of test data for dataset: '{self.dataset}' is {trainSize_total}")
            return False
        n = self.compression

        def reshape(image, label):
            image = tf.image.resize(image, [n, n], method='gaussian', antialias=True)
            image = tf.cast(image, tf.float32) / 127.5 - 1
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

        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)
        self.test_X = np.array(test_X)
        self.test_y = np.array(test_y)
        return True

    def __getStates(self, img):
        e = Evolve(img, self.t_total, self.pc_real, self.pc_imag, self.pq_real, self.pq_imag, self.N, self.g, self.nsteps, self.intervals)
        return e.states()
    def __multiprocess(self, train_imgs, test_imgs, max_pool=10):
        if max_pool <= 1:
            train_X_states = list(map(self.__getStates, tqdm(train_imgs)))
            test_X_states = list(map(self.__getStates, tqdm(test_imgs)))
        else:
            with Pool(max_pool) as p:
                train_X_states = list(
                    tqdm(
                        p.imap(self.__getStates,train_imgs),
                        total=self.trainSize

                    )
                )
                test_X_states = list(
                    tqdm(
                        p.imap(self.__getStates,test_imgs),
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
        self.train_X_states = train_X_states
        self.test_X_states = test_X_states
        self.trained = True
        return self

    def __wigner(self, states):
        w = []
        for state in states:
            w += list(wigner(state, self.xvec, self.yvec).flatten())
        return np.real(w)

    def __rho(self, states):
        r = np.array([])
        for rho in states:
            rho.tidyup()
            diag = np.real(rho.diag()[:-1])
            x = rho.full()
            y = x[np.triu_indices_from(x,k=1)]
            r = np.concatenate((r, diag, y.real, y.imag))
        return r
    def __disp(self, states):
        r = []
        for rho in states:
            rho.tidyup()
            for D in self.displacements:
                m = (self.P*D.dag()*rho*D).tr()
                r.append(m)
        return np.real(r)

    def process(self, xRange=(-2.5, 2), yRange=(-2, 4.5), res=30):
        if not self.trained:
            print("Unable to process. Data has not been trained")
            return
        xvec = np.linspace(*xRange, res)
        yvec = np.linspace(*yRange, res)
        self.xvec = xvec
        self.yvec = yvec
        print("Vectorizing quantum states...")
        self.train_X_processed = list(map(self.__wigner, tqdm(self.train_X_states)))
        self.test_X_processed = list(map(self.__wigner, tqdm(self.test_X_states)))
        self.processed = True
        return self
    def process_rho(self):
        if not self.trained:
            print("Unable to process. Data has not been trained")
            return self
        print("Vectorizing quantum states...")
        self.train_X_processed = list(map(self.__rho, tqdm(self.train_X_states)))
        self.test_X_processed = list(map(self.__rho, tqdm(self.test_X_states)))
        self.processed = True
        return self
    def process_disp(self):
        if not self.trained:
            print("Unable to process. Data has not been trained")
            return self
        print("Vectorizing quantum states...")
        self.train_X_processed = list(map(self.__disp, tqdm(self.train_X_states)))
        self.test_X_processed = list(map(self.__disp, tqdm(self.test_X_states)))
        self.processed = True
        return self

    def scoreR(self):
        mq = []
        for i in range(5, 26, 3):
            clf = RidgeClassifier(alpha=pow(10, -i))
            clf.fit(self.train_X_processed, self.train_y)
            q_score = clf.score(self.test_X_processed, self.test_y)
            mq.append(q_score)
        clf = RidgeClassifier(alpha=0)
        clf.fit(self.train_X_processed, self.train_y)
        q_score = clf.score(self.test_X_processed, self.test_y)
        mq.append(q_score)
        return max(mq)
    def scoreSelf(self):
        mq = []
        for i in range(5, 26, 3):
            clf = RidgeClassifier(alpha=pow(10, -i))
            clf.fit(self.train_X_processed, self.train_y)
            q_score = clf.score(self.train_X_processed, self.train_y)
            mq.append(q_score)
        clf = RidgeClassifier(alpha=0)
        clf.fit(self.train_X_processed, self.train_y)
        q_score = clf.score(self.train_X_processed, self.train_y)
        mq.append(q_score)
        return max(mq)
    def score(self, n_regressors=("LogCV",), q_regressors=("Ridge",), n_alpha=0.1, q_alpha=0):
        if not self.processed:
            print("Unable to score. Data has not been processed")
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
                case "LogCV":
                    clf = LogisticRegressionCV(max_iter=1000)
                case "Ridge":
                    clf = RidgeClassifier(alpha=q_alpha)
                case _:
                    print("Regression type not valid")
                    continue
            clf.fit(self.train_X_processed, self.train_y)
            q_score = clf.score(self.test_X_processed, self.test_y)
            print(f"{reg_dict[r]} (With Quantum processing) yields: {q_score}")


