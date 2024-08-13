
import numpy as np
import math
import matplotlib.pyplot as plt
from qutip import *
import scipy.io
from scipy.linalg import sqrtm
# from numba import jit
import time

start_time = time.time()  # checking how long the code takes


# generate gellman matrices
def gellman(D):
    n_para = D ** 2 - 1
    G = np.zeros([n_para, D, D], dtype=np.complex_)

    def Eone(i, j):
        if i >= D or j >= D:
            print("Invalid indices. Indices must be less than matrix dimension.")
            return None
        matrix = [[0] * D for _ in range(D)]
        matrix[i][j] = 1
        return np.array(matrix)

    ct = 0
    for k in range(D):
        for j in range(k):
            G[ct] = Eone(j, k) + Eone(k, j)
            ct += 1
            G[ct] = -1j * (Eone(j, k) - Eone(k, j))
            ct += 1
    for l in range(D - 1):
        K = np.zeros([D, D], dtype=np.complex_)
        for j in range(l + 1):
            K += Eone(j, j)
        G[ct] = np.sqrt(2 / ((l + 1) * (l + 2))) * (K - (l + 1) * Eone(l + 1, l + 1))
        ct += 1
    return G


def reimdiagobs(D, full=False):
    n_para = D ** 2 - 1
    G = np.zeros([n_para, D, D], dtype=np.complex_)

    def Eone(i, j):
        if i >= D or j >= D:
            print("Invalid indices. Indices must be less than matrix dimension.")
            return None
        matrix = [[0] * D for _ in range(D)]
        matrix[i][j] = 1
        return np.array(matrix)

    ct = 0
    for l in range(D - 1):
        G[ct] = Eone(l, l)
        ct += 1
    for k in range(D):
        for j in range(k):
            G[ct] = (Eone(j, k) + Eone(k, j)) / 2
            ct += 1
            G[ct] = -(-1j * (Eone(j, k) - Eone(k, j))) / 2
            ct += 1
    if full:
        Gfull = np.zeros([n_para + 1, D, D], dtype=np.complex_)
        Gfull[:-1, :, :] = G
        Gfull[-1] = Eone(D - 1, D - 1)
        return Gfull
    else:
        return G


def QN_regression(X_R, Y_R, lamb):
    Error = 0
    Ntr = len(X_R[0, :])
    RM = len(X_R[:, 0]) - 1
    nD = len(Y_R[:, 0])

    a1 = np.matmul(Y_R, np.transpose(X_R))
    b1 = np.linalg.inv(np.matmul(X_R, np.transpose(X_R)) + lamb * np.eye(RM + 1))
    beta = np.matmul(a1, b1)

    Er = np.zeros([Ntr, 1])
    for k in np.arange(0, Ntr):
        Y_id = Y_R[:, k]  # targets
        X_ob = X_R[:, k]  # 1 and observables
        Y_est = np.matmul(beta, X_ob)  # estimations
        Er[k] = np.sum(np.absolute(Y_id - Y_est)) / nD
    Error = np.mean(Er)

    return Error, beta


def PSD_rho(rho):  # -> pos semi definite rho, given hermitian with trace 1
    d = rho.shape[0]
    w, v = np.linalg.eig(rho)
    idx = np.argsort(w)[::-1]  # reverse, now from largest to smallest
    w = w[idx]
    v = v[:, idx]

    la = 0 * w  # to store eigenvalues of new state
    a = 0  # accumulator
    i = d - 1  # index

    while w[i] + a / (i + 1) < 0:
        la[i] = 0
        a += w[i]
        i += -1

    for k in np.arange(0, i + 1):
        la[k] = w[k] + a / (i + 1)

    rho_f = 0 * rho  # store final density matrix
    for x in np.arange(0, len(la)):
        rho_f = rho_f + (la[x] * Qobj(v[:, x]) * Qobj(v[:, x]).dag()).full()

    return Qobj(rho_f)


def PSD_rho_cn(rho):  # -> pos semi definite rho, given hermitian with trace 1
    d = rho.shape[0]
    w, v = np.linalg.eig(rho)
    idx = np.argsort(w)[::-1]  # reverse, now from largest to smallest
    w = w[idx]
    v = v[:, idx]

    cn = 0
    if np.min(w) < 0:
        cn = np.min(w)

    la = 0 * w  # to store eigenvalues of new state
    a = 0  # accumulator
    i = d - 1  # index

    while w[i] + a / (i + 1) < 0:
        la[i] = 0
        a += w[i]
        i += -1

    for k in np.arange(0, i + 1):
        la[k] = w[k] + a / (i + 1)

    rho_f = 0 * rho.full()  # store final density matrix
    for x in np.arange(0, len(la)):
        rho_f = rho_f + (la[x] * Qobj(v[:, x]) * Qobj(v[:, x]).dag()).full()

    return Qobj(rho_f), cn.real


def tr1_rho(M, W):
    d = int(np.sqrt(M.shape[1]))
    Id = qeye(d)
    Idvec = (np.transpose(Id)).reshape((d ** 2, 1))  # identity in vec form
    MM = np.zeros([d ** 2 + 1, d ** 2 + 1], dtype=complex)
    XX = np.zeros([d ** 2 + 1, 1], dtype=complex)
    MM[:d ** 2, :d ** 2] = M.dag() * M
    MM[:d ** 2, d ** 2] = np.transpose(Idvec)  # apparently it works like this
    MM[d ** 2, :d ** 2] = np.transpose(Idvec)
    MM[d ** 2, d ** 2] = 0
    XX[:d ** 2, 0] = np.transpose(M.dag() * W)  # apparently it works like this
    XX[d ** 2, 0] = 1
    YY = np.matmul(np.linalg.inv(MM), XX)
    rvec_est = YY[:d ** 2, 0]
    return rvec_est


def Bysn_rho(numSamp, N, rho_tar, rho_est):
    # loop parameters
    THIN = np.array([2 ** 7])  # 2**np.arange(8)#
    samplers = 1

    # inputs
    # numSamp = 2**10
    alpha = 1

    Mb = 500
    r = 1.1

    # data
    # mat_data = scipy.io.loadmat(r"C:\Users\tanju\Dropbox\NTU Grad\Research\MATLAB codes\BayesQuanTom-master\Simulated Dataset\simData_L=0.85_d=3.mat")
    # mat_data = scipy.io.loadmat("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/MATLAB codes/BayesQuanTom-master/Simulated Dataset/simData_L=0.85_d=3.mat")
    # psi0 = Qobj(mat_data['psi0'])
    # D = psi0.shape[0]
    # rho_tar = ket2dm(psi0)#rand_dm(cdim)
    D = rho_tar.shape[0]
    numParam = 2 * D ** 2 + D

    # N = np.sum(mat_data['counts'])#14400#total no of events
    sigma = 1 / np.sqrt(N)
    # rhoLS = Qobj(mat_data['rhoLS'])#(rho_tar + 0.05*qeye(cdim)).unit()#rho_est
    rhoLS = rho_est
    # print(f"fidelity for LS is {fidelity(rho_tar, rhoLS)**2}")
    rhoLSvec = Qobj(rhoLS.full().reshape([D ** 2, 1]))  # rho_est col

    # sampling loop
    Fmean = np.zeros([samplers, len(THIN)])
    Fstd = np.zeros([samplers, len(THIN)])
    samplerTime = np.zeros([samplers, len(THIN)])

    # param to rho col function
    def paramToRhoCol(par):
        # D = 9
        # par = np.random.random(171)
        Xr = np.transpose(par[0:D ** 2].reshape([D, D]))  # to be consistent with MATLAB code reshaping
        Xi = np.transpose(par[D ** 2:2 * D ** 2].reshape([D, D]))
        X = Xr + 1j * Xi  # matrix of column vectors (not normalised)

        NORM = np.linalg.norm(X, axis=0)  # norm of each column
        W = X / NORM  # normalise each column

        Y = par[2 * D ** 2:]  # projector weights
        gamma = Y / np.sum(Y)  # normalise

        rho = Qobj(W) * Qobj(np.diag(gamma)) * Qobj(W).dag()
        z = Qobj(rho.full().reshape([D ** 2, 1]))
        return z

    rho_BME = rhoLS * 0
    # the loop
    for k in range(len(THIN)):
        for m in range(samplers):
            param0 = np.zeros(numParam)
            Fest = np.zeros([numSamp, 1])

            np.random.seed(int(time.time() - start_time))  # change the seed based on time to ensure good random
            param0[0:2 * D ** 2] = np.random.randn(2 * D ** 2)  # initial seed
            param0[2 * D ** 2:] = np.random.gamma(alpha, scale=1, size=D)

            beta1 = 0.1  # initial parameters for stepsize
            beta2 = 0.1
            acc = 0  # counter of acceptances

            # initial point
            x = param0
            rhoX = paramToRhoCol(x)
            logX = -1 / (2 * sigma ** 2) * np.linalg.norm(rhoX - rhoLSvec) ** 2 + np.sum(
                alpha * np.log(x[2 * D ** 2:]) - x[2 * D ** 2:])

            # pCN loop
            tt = time.time()
            for j in range(numSamp * THIN[k]):
                # proposed update parameters:
                newGauss = np.sqrt(1 - beta1 ** 2) * x[0:2 * D ** 2] + beta1 * np.random.randn(2 * D ** 2)
                newGamma = x[2 * D ** 2:] * np.exp(beta2 * np.random.randn(D))
                y = np.concatenate((newGauss, newGamma))

                rhoY = paramToRhoCol(y)
                logY = -1 / (2 * sigma ** 2) * np.linalg.norm(rhoY - rhoLSvec) ** 2 + np.sum(
                    alpha * np.log(y[2 * D ** 2:]) - y[2 * D ** 2:])

                if np.log(np.random.random(1)[0]) < logY - logX:
                    x = y
                    logX = logY
                    acc += 1

                if j % Mb == 0:  # stepsize adaptation
                    rat = acc / Mb  # estimate acceptance probability, and keep near optimal 0.234
                    if rat > 0.3:
                        beta1 *= r
                        beta2 *= r
                    elif rat < 0.1:
                        beta1 /= r
                        beta2 /= r
                    acc = 0

                if j % THIN[k] == 0:
                    rhoAsVec = paramToRhoCol(x)
                    rhoEst = Qobj(rhoAsVec.full().reshape([D, D]))
                    rho_BME += rhoEst
                    Fest[int(j / THIN[k])] = fidelity(rho_tar, rhoEst) ** 2

            samplerTime[m, k] = time.time() - tt

            # quantities of interest
            Fmean[m, k] = np.mean(Fest)
            Fstd[m, k] = np.std(Fest, ddof=1)
    return Fmean, Fstd, rho_BME / numSamp  # this gives rho_BME


def Bysn_rho_v2(numSamp, N, rho_tar, rhoLS):
    def fidelity_f(rho1, rho2):
        sqrt_rho1 = sqrtm(rho1)

        term = sqrt_rho1 @ rho2 @ sqrt_rho1
        trace_term = np.trace(sqrtm(term))

        fidelity_value = np.abs(trace_term) ** 2

        return fidelity_value

    # loop parameters
    THIN = np.array([2 ** 7])  # 2**np.arange(8)#np.array([2**7])#
    samplers = 1

    # inputs
    # numSamp = 2**10
    alpha = 1

    Mb = 500
    r = 1.1

    # data
    # d = cdim
    # D = d
    # mat_data = scipy.io.loadmat(r"C:\Users\tanju\Dropbox\NTU Grad\Research\MATLAB codes\BayesQuanTom-master\Simulated Dataset\simData_L=0.85_d=3.mat")
    # mat_data = scipy.io.loadmat("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/MATLAB codes/BayesQuanTom-master/Simulated Dataset/simData_L=0.85_d=3.mat")
    # psi0 = mat_data['psi0']
    # D = psi0.shape[0]
    # rho_tar = psi0 @ psi0.transpose().conj()#rand_dm(cdim)
    D = rho_tar.shape[0]
    numParam = 2 * D ** 2 + D

    # N = np.sum(mat_data['counts'])#14400#total no of events
    sigma = 1 / np.sqrt(N)
    # rhoLS = mat_data['rhoLS']#(rho_tar + 0.05*qeye(cdim)).unit()#rho_est
    # print(f"fidelity for LS is {fidelity(rho_tar, rhoLS)}")
    rhoLSvec = rhoLS.reshape([D ** 2, 1])  # rho_est col

    # sampling loop
    Fmean = np.zeros([samplers, len(THIN)])
    Fstd = np.zeros([samplers, len(THIN)])
    samplerTime = np.zeros([samplers, len(THIN)])

    # param to rho col function
    def paramToRhoCol(par):
        # D = 9
        # par = np.random.random(171)
        Xr = np.transpose(par[0:D ** 2].reshape([D, D]))  # to be consistent with MATLAB code reshaping
        Xi = np.transpose(par[D ** 2:2 * D ** 2].reshape([D, D]))
        X = Xr + 1j * Xi  # matrix of column vectors (not normalised)

        NORM = np.linalg.norm(X, axis=0)  # norm of each column
        W = X / NORM  # normalise each column

        Y = par[2 * D ** 2:]  # projector weights
        gamma = Y / np.sum(Y)  # normalise

        rho = W @ np.diag(gamma) @ W.transpose().conj()
        z = rho.reshape([D ** 2, 1])
        return z

    rho_BME = rhoLS * 0
    # the loop
    for k in range(len(THIN)):
        for m in range(samplers):
            param0 = np.zeros(numParam)
            Fest = np.zeros([numSamp, 1])

            np.random.seed(int(time.time() - start_time))  # change the seed based on time to ensure good random
            param0[0:2 * D ** 2] = np.random.randn(2 * D ** 2)  # initial seed
            param0[2 * D ** 2:] = np.random.gamma(alpha, scale=1, size=D)

            beta1 = 0.1  # initial parameters for stepsize
            beta2 = 0.1
            acc = 0  # counter of acceptances

            # initial point
            x = param0
            rhoX = paramToRhoCol(x)
            logX = -1 / (2 * sigma ** 2) * np.linalg.norm(rhoX - rhoLSvec) ** 2 + np.sum(
                alpha * np.log(x[2 * D ** 2:]) - x[2 * D ** 2:])

            # pCN loop
            tt = time.time()
            for j in range(numSamp * THIN[k]):
                # proposed update parameters:
                newGauss = np.sqrt(1 - beta1 ** 2) * x[0:2 * D ** 2] + beta1 * np.random.randn(2 * D ** 2)
                newGamma = x[2 * D ** 2:] * np.exp(beta2 * np.random.randn(D))
                y = np.concatenate((newGauss, newGamma))

                rhoY = paramToRhoCol(y)
                logY = -1 / (2 * sigma ** 2) * np.linalg.norm(rhoY - rhoLSvec) ** 2 + np.sum(
                    alpha * np.log(y[2 * D ** 2:]) - y[2 * D ** 2:])

                if np.log(np.random.random(1)[0]) < logY - logX:
                    x = y
                    logX = logY
                    acc += 1

                if j % Mb == 0:  # stepsize adaptation
                    rat = acc / Mb  # estimate acceptance probability, and keep near optimal 0.234
                    if rat > 0.3:
                        beta1 *= r
                        beta2 *= r
                    elif rat < 0.1:
                        beta1 /= r
                        beta2 /= r
                    acc = 0

                if j % THIN[k] == 0:
                    rhoAsVec = paramToRhoCol(x)
                    rhoEst = rhoAsVec.reshape([D, D])
                    rho_BME += rhoEst
                    Fest[int(j / THIN[k])] = fidelity_f(rho_tar, rhoEst)
            samplerTime[m, k] = time.time() - tt

            # quantities of interest
            Fmean[m, k] = np.mean(Fest)
            Fstd[m, k] = np.std(Fest, ddof=1)
    return Fmean[0, 0], Fstd[0, 0], rho_BME / numSamp  # this gives rho_BME

def Bysn_rho_v3(numSamp, N, rhoLS):

    # loop parameters
    THIN = np.array([2 ** 7])  # 2**np.arange(8)#np.array([2**7])#
    samplers = 1

    # inputs
    # numSamp = 2**10
    alpha = 1

    Mb = 500
    r = 1.1

    # data
    # d = cdim
    # D = d
    # mat_data = scipy.io.loadmat(r"C:\Users\tanju\Dropbox\NTU Grad\Research\MATLAB codes\BayesQuanTom-master\Simulated Dataset\simData_L=0.85_d=3.mat")
    # mat_data = scipy.io.loadmat("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/MATLAB codes/BayesQuanTom-master/Simulated Dataset/simData_L=0.85_d=3.mat")
    # psi0 = mat_data['psi0']
    # D = psi0.shape[0]
    # rho_tar = psi0 @ psi0.transpose().conj()#rand_dm(cdim)
    D = rhoLS.shape[0]
    numParam = 2 * D ** 2 + D

    # N = np.sum(mat_data['counts'])#14400#total no of events
    sigma = 1 / np.sqrt(N)
    # rhoLS = mat_data['rhoLS']#(rho_tar + 0.05*qeye(cdim)).unit()#rho_est
    # print(f"fidelity for LS is {fidelity(rho_tar, rhoLS)}")
    rhoLSvec = rhoLS.reshape([D ** 2, 1])  # rho_est col


    # param to rho col function
    def paramToRhoCol(par):
        # D = 9
        # par = np.random.random(171)
        Xr = np.transpose(par[0:D ** 2].reshape([D, D]))  # to be consistent with MATLAB code reshaping
        Xi = np.transpose(par[D ** 2:2 * D ** 2].reshape([D, D]))
        X = Xr + 1j * Xi  # matrix of column vectors (not normalised)

        NORM = np.linalg.norm(X, axis=0)  # norm of each column
        W = X / NORM  # normalise each column

        Y = par[2 * D ** 2:]  # projector weights
        gamma = Y / np.sum(Y)  # normalise

        rho = W @ np.diag(gamma) @ W.transpose().conj()
        z = rho.reshape([D ** 2, 1])
        return z

    rho_BME = rhoLS * 0
    # the loop
    for k in range(len(THIN)):
        for m in range(samplers):
            param0 = np.zeros(numParam)
            Fest = np.zeros([numSamp, 1])

            np.random.seed(int(time.time() - start_time))  # change the seed based on time to ensure good random
            param0[0:2 * D ** 2] = np.random.randn(2 * D ** 2)  # initial seed
            param0[2 * D ** 2:] = np.random.gamma(alpha, scale=1, size=D)

            beta1 = 0.1  # initial parameters for stepsize
            beta2 = 0.1
            acc = 0  # counter of acceptances

            # initial point
            x = param0
            rhoX = paramToRhoCol(x)
            logX = -1 / (2 * sigma ** 2) * np.linalg.norm(rhoX - rhoLSvec) ** 2 + np.sum(
                alpha * np.log(x[2 * D ** 2:]) - x[2 * D ** 2:])

            # pCN loop
            tt = time.time()
            for j in range(numSamp * THIN[k]):
                # proposed update parameters:
                newGauss = np.sqrt(1 - beta1 ** 2) * x[0:2 * D ** 2] + beta1 * np.random.randn(2 * D ** 2)
                newGamma = x[2 * D ** 2:] * np.exp(beta2 * np.random.randn(D))
                y = np.concatenate((newGauss, newGamma))

                rhoY = paramToRhoCol(y)
                logY = -1 / (2 * sigma ** 2) * np.linalg.norm(rhoY - rhoLSvec) ** 2 + np.sum(
                    alpha * np.log(y[2 * D ** 2:]) - y[2 * D ** 2:])

                if np.log(np.random.random(1)[0]) < logY - logX:
                    x = y
                    logX = logY
                    acc += 1

                if j % Mb == 0:  # stepsize adaptation
                    rat = acc / Mb  # estimate acceptance probability, and keep near optimal 0.234
                    if rat > 0.3:
                        beta1 *= r
                        beta2 *= r
                    elif rat < 0.1:
                        beta1 /= r
                        beta2 /= r
                    acc = 0

                if j % THIN[k] == 0:
                    rhoAsVec = paramToRhoCol(x)
                    rhoEst = rhoAsVec.reshape([D, D])
                    rho_BME += rhoEst
    return rho_BME / numSamp  # this gives rho_BME