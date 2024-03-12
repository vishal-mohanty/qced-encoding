import os
import numpy as np
from tqdm import tqdm
from src.encoding.evolution import Evolve
from multiprocess.pool import Pool
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier, LogisticRegression
from qutip import wigner

class Classify:
    cur_path = os.path.dirname(__file__)
    def __init__(self, compression=4,
                 t_total=12, t_steps=50, coef_pc=0.25, coef_pq=0.25, N=20, g=1, dataset="mnist"):
        self.compression = compression
        self.t_total = t_total
        self.t_steps = t_steps
        self.coef_pc = coef_pc
        self.coef_pq = coef_pq
        self.N = N
        self.g = g
        self.dataset = dataset
        self.loaded = True
        self.__loadData(dataset)
        self.trained = False
        self.processed = False
    def __loadData(self, dataset):
        n = self.compression
        match dataset:
            case "mnist":
                save_path = os.path.relpath(f"data\\datasets\\mnist_{self.compression}.npz", self.cur_path)

            case "fashion":
                save_path = os.path.relpath(f"data\\datasets\\fmnist_{self.compression}.npz", self.cur_path)
            case _:
                print("No such dataset exists")
                self.loaded = False
                return
        if os.path.isfile(save_path):
            data = np.load(save_path)
            self.train_X, self.train_y, self.test_X, self.test_y = data['train_X'], data['train_y'], data['test_X'], data['test_y']
        else:
            match dataset:
                case "fashion":
                    from tensorflow.keras.datasets import fashion_mnist as mnist
                case "mnist":
                    from tensorflow.keras.datasets import mnist
                case _:
                    from tensorflow.keras.datasets import mnist
            import tensorflow as tf
            (train_X, self.train_y), (test_X, self.test_y) = mnist.load_data()
            trainSize_total = len(train_X)
            testSize_total = len(test_X)
            train_X = tf.image.resize(train_X[..., tf.newaxis], [n,n], method='gaussian', antialias=True)
            test_X = tf.image.resize(test_X[..., tf.newaxis], [n,n], method='gaussian', antialias=True)
            self.train_X = (np.reshape(train_X, (trainSize_total, n*n))-128)/128
            self.test_X = (np.reshape(test_X, (testSize_total, n*n))-128)/128
            with open(save_path, "wb") as f:
                np.savez(f, train_X=self.train_X, train_y=self.train_y, test_X=self.test_X, test_y=self.test_y)

    def __getStates(self, img):
        e = Evolve(img, self.t_total, self.t_steps, self.coef_pc, self.coef_pq, self.N, self.g)
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
        if not self.loaded:
            print("Unable to train. No Datasets Loaded.")
            return
        train_imgs = self.train_X[:trainSize]
        test_imgs = self.test_X[:testSize]
        self.trainSize = trainSize
        self.testSize = testSize
        self.train_X_states, self.test_X_states = self.__multiprocess(train_imgs, test_imgs, max_pool=max_pool)
        self.trained = True
        return self

    def __state(self, states, p):
        return states[int(p*(self.t_steps-1))]

    def __wigner(self, states):
        w = []
        for i in range(1, self.copies+1):
            w += list(wigner(self.__state(states, i / self.copies), self.xvec, self.yvec).flatten())
        return np.real(w)

    def process(self, copies=8,  xRange=(-2.5, 2), yRange=(-2, 4.5), res=30):
        if not self.trained:
            print("Unable to process. Data has not been trained")
            return
        xvec = np.linspace(*xRange, res)
        yvec = np.linspace(*yRange, res)
        self.copies = copies
        self.xvec = xvec
        self.yvec = yvec
        self.train_X_processed = list(map(self.__wigner, tqdm(self.train_X_states)))
        self.test_X_processed = list(map(self.__wigner, tqdm(self.test_X_states)))
        self.processed = True
        return self
    def score(self, n_regressors=("LogCV",), q_regressors=("Ridge",), n_alpha=0.1, q_alpha=0):
        if not self.processed:
            print("Unable to score. Data has not been processed")
            return
        trainRange = slice(0, self.trainSize)
        testRange = slice(0, self.testSize)
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
            clf.fit(self.train_X[trainRange], self.train_y[trainRange])
            score = clf.score(self.test_X[testRange], self.test_y[testRange])
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
            clf.fit(self.train_X_processed, self.train_y[trainRange])
            q_score = clf.score(self.test_X_processed, self.test_y[testRange])
            print(f"{reg_dict[r]} (With Quantum processing) yields: {q_score}")


