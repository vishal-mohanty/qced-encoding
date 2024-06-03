import numpy as np
from qutip import Qobj, destroy, displace, fock_dm
from src.TK_basics import PSD_rho
from tqdm import tqdm
from multiprocess.pool import Pool
from sklearn.linear_model import RidgeClassifier
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
def vectorize(y):
    I = np.eye(10)
    arr = []
    for i in y: arr.append(I[i])
    return np.column_stack(arr)
def lreg(X, y):
    x = X.T
    X1 = np.vstack((np.ones(len(x[0])), x))
    cM = vectorize(y)@np.linalg.pinv(X1)
    c = cM[:, 0]
    c = c[..., None]
    M = cM[:, 1:]
    return M, c

def vec2rho(vec, intervals):
    n = len(vec)/intervals
    if int(np.sqrt(n+1)) != np.sqrt(n+1):
        print("vector is not of length d^2 - 1")
        return None
    d = int(np.sqrt(n+1))
    results = []
    vecs = np.split(vec, intervals)
    for v in vecs:
        k = 0
        s = 0
        rho = np.zeros(shape=(d, d), dtype=complex)
        for i, j in zip(*np.triu_indices(d)):
            if i == j == d - 1: continue
            rho[i, j] = v[k]
            k += 1
            if i != j:
                rho[i, j] += v[k]*1j
                rho[j, i] = np.conj(rho[i, j])
                k += 1
            else:
                s += rho[i, j]
        rho[d-1, d-1] = 1-s
        results.append(rho)
    return results

def rho2vec(rho):
    if type(rho) is list:
        rhos = []
        for r in rho:
            if type(r) is Qobj:
                rhos.append(r.full())
            else:
                rhos.append(r)
    elif type(rho) is Qobj:
        rhos = [rho.full()]
    else:
        rhos = [rho]
    d = len(rhos[0])
    vec = []
    for rho in rhos:
        for i, j in zip(*np.triu_indices(d)):
            if i == j == d - 1:
                continue
            vec.append(rho[i, j].real)
            if i != j:
                vec.append(rho[i, j].imag)
    return np.array(vec)

def getEp(cdim, dm, method):
    if method == "par" or method == "parity" or method == "par trace" or method == "parity trace":
        displacements = np.load(f'..\\data\\disps\\disps2.npy')
        E = []
        a = destroy(cdim)
        P = (1j*np.pi*a.dag()*a).expm()
        for alpha in displacements:
            D = displace(cdim, alpha)
            E.append(np.hstack(D.dag()*P*D)[0])
    else:
        displacements = np.load(f'..\\data\\disps\\pns_disps{dm}.npy')
        E = []
        proj = fock_dm(cdim, dm-1)
        for alpha in displacements:
            D = displace(cdim, alpha)
            E.append(np.hstack(D.dag()*proj*D)[0])
    return np.linalg.pinv(np.array(E))

def getrho(i):

    split = i[0]
    vecs_g = np.split(i[1], split)
    vecs_e = np.split(i[2], split)
    ps_g = np.split(i[3], split)
    ps_e = np.split(i[4], split)
    Ep = i[5]
    cdim = i[6]
    vec = []
    for vec_g, vec_e, pg, pe in zip(vecs_g, vecs_e, ps_g, ps_e):
        rho_g_vec = Ep@vec_g
        rho_e_vec = Ep@vec_e
        rho_g = np.hstack([c[..., None] for c in np.split(rho_g_vec,cdim)])
        rho_g = PSD_rho(Qobj(rho_g[:cdim,:cdim]).unit())
        rho_e = np.hstack([c[..., None] for c in np.split(rho_e_vec,cdim)])
        rho_e = PSD_rho(Qobj(rho_e[:cdim,:cdim]).unit())
        pg = pg/(pg+pe)
        pe = pe/(pg+pe)
        rho = (pg[0]*rho_g + pe[0]*rho_e).unit()
        vec += list(rho2vec(PSD_rho(rho)))
    return vec


def getRho(X_g, X_e, p_g, p_e, cdim, split, method):
    dm = int(np.sqrt((X_g.shape[1] + split) / split))
    Ep = getEp(cdim, dm, method)
    max_pool = 13
    cdims = [cdim for _ in X_g]
    Eps = [Ep for _ in X_g]
    splits = [split for _ in X_g]
    items = list(zip(splits, X_g, X_e, p_g, p_e, Eps, cdims))
    with Pool(max_pool) as p:
        X_rho = list(
            tqdm(
                p.imap(getrho, items),
                total=len(items)

            )
        )
    return np.array(X_rho)

def getrho2(i):
    split = i[0]
    vecs = np.split(i[1], split)
    Ep = i[2]
    cdim = i[3]
    vecz = []
    for vec in vecs:
        rho_vec = Ep@vec
        rho = np.hstack([c[..., None] for c in np.split(rho_vec,cdim)])
        rho = Qobj(rho[:cdim,:cdim]).unit()

        vecz += list(rho2vec(PSD_rho(rho)))
    return vecz

def getRho2(X, cdim, split, method):
    dm = int(np.sqrt((X.shape[1] + split) / split))
    Ep = getEp(cdim, dm, method)
    max_pool = 13
    cdims = [cdim for _ in X]
    Eps = [Ep for _ in X]
    splits = [split for _ in X]
    items = list(zip(splits, X, Eps, cdims))
    with Pool(max_pool) as p:
        X_rho = list(
            tqdm(
                p.imap(getrho2, items),
                total=len(items)

            )
        )
    return np.array(X_rho)
def std(data, n, iterations=1):
    split = data.intervals
    cdim = data.cdim
    if data.method == "1qubit":
        trainX_g = data.train_X_g
        trainX_e = data.train_X_e
        trainX_pg = data.train_X_p_g
        trainX_pe = data.train_X_p_e
        testX_g = data.test_X_g
        testX_e = data.test_X_e
        testX_pg = data.test_X_p_g
        testX_pe = data.test_X_p_e
        trainy = data.train_y
        testy = data.test_y
        tx = data.train_X
        ttx = data.test_X
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

            trainX_est = getRho(trainX_g_est, trainX_e_est, trainX_pg_est, trainX_pe_est, cdim, split, data.measurement)
            testX_est = getRho(testX_g_est, testX_e_est, testX_pg_est, testX_pe_est, cdim, split, data.measurement)

            clf = RidgeClassifier(alpha=0)
            clf.fit(np.hstack((tx, trainX_est)), trainy)
            results.append(clf.score(np.hstack((ttx, testX_est)), testy))
        results = np.array(results)
        if n < 1: print(f"std={n}: ", results.mean())
        else: print(f"samples={n}: ", results.mean())
        return results.mean()
    elif data.method == "2qubit":
        trainX = data.train_X_processed
        testX = data.test_X_processed
        trainy = data.train_y
        testy = data.test_y
        tx = data.train_X
        ttx = data.test_X
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

            trainX_est = getRho2(trainX_est, cdim, split, data.measurement)
            testX_est = getRho2(testX_est, cdim, split, data.measurement)

            clf = RidgeClassifier(alpha=0)
            clf.fit(np.hstack((tx, trainX_est)), trainy)
            results.append(clf.score(np.hstack((ttx, testX_est)), testy))
        results = np.array(results)
        if n < 1: print(f"std={n}: ", results.mean())
        else: print(f"samples={n}: ", results.mean())
        return results.mean()
