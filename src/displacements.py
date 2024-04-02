import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.special import genlaguerre
from math import sqrt , factorial
from numpy.linalg import cond , svd
from scipy.optimize import fmin , check_grad , minimize
from IPython.display import display , clear_output
import time

# Number of photons
FD = 11
# Number of displacements
n_disps = FD**2# + 30

def wigner_mat_and_grad(disps, FD):
    ND = len(disps)#no of exp points/displacements, equals n_disps
    wig_tens = np.zeros((ND, FD, FD), dtype=complex)
    grad_mat_r = np.zeros((ND, FD, FD), dtype=complex)
    grad_mat_i = np.zeros((ND, FD, FD), dtype=complex)

    B = 4*np.abs(disps)**2
    pf = (2/np.pi)*np.exp(-B/2)
    for m in range(FD):
        x = pf * np.real((-1) ** m * genlaguerre(m, 0)(B))
        term_r = -4 * disps.real * x
        term_i = -4 * disps.imag * x

        if m > 0:
            y = 8 * pf * (-1)**(m-1) * genlaguerre(m-1, 1)(B)
            term_r += disps.real * y
            term_i += disps.imag * y
        wig_tens[:, m, m] = x
        grad_mat_r [:, m, m] = term_r
        grad_mat_i [:, m, m] = term_i

        for n in range(m+1, FD):
            pf_nm = sqrt(factorial(m)/float(factorial(n)))
            x = pf * pf_nm * (-1)**m * 2 * (2*disps)**(n-m-1) * genlaguerre(m, n-m)(B)
            term_r = ((n - m) - 4*disps.real*disps) * x
            term_i = (1j * (n - m) - 4*disps.imag*disps) * x
            if m > 0:
                y = 8 * pf * pf_nm * (-1)**(m-1) * (2*disps)**(n-m) *\
                    genlaguerre(m-1, n-m+1)(B)
                term_r += disps.real * y
                term_i += disps.imag * y
            wig_tens[:, m, n] = disps * x
            wig_tens[:, n, m] = (disps * x).conj()
            grad_mat_r[:, m, n] = term_r
            grad_mat_r[:, n, m] = term_r.conjugate()
            grad_mat_i[:, m, n] = term_i
            grad_mat_i[:, n, m] = term_i.conjugate()
    return (wig_tens.reshape((ND, FD**2)), grad_mat_r.reshape((ND, FD**2)), grad_mat_i.reshape((ND, FD**2)))

def cost_and_grad(r_disps):
    N = len(r_disps)
    c_disps = r_disps[:int(N/2)] + 1j*r_disps[int(N/2):]#now complex displacements
    M, dM_rs, dM_is = wigner_mat_and_grad (c_disps , FD)
    U, S, Vd = svd(M)
    NS = len(Vd)
    cn = S[0] / S[-1]
    dS_r = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_rs , Vd.conj().T).real
    dS_i = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_is , Vd.conj().T).real
    grad_cn_r = (dS_r[0]*S[-1] - S[0]*dS_r[-1])/(S[-1]**2)
    grad_cn_i = (dS_i[0]*S[-1] - S[0]*dS_i[-1])/(S[-1]**2)
    return cn, np.concatenate(( grad_cn_r , grad_cn_i))

best_cost = float('inf')
f, ax = plt.subplots(figsize=(5, 5))

def wrap_cost(disps):
    global best_cost
    cost , grad = cost_and_grad(disps)
    best_cost = min(cost , best_cost)
    ax.clear()
    ax.plot(disps[:n_disps], disps[n_disps:], 'k.')#ok
    ax.set_title('Condition Number = %.1f' % (cost,))
    ax.set_xlabel('Re'+r'$(\alpha)$')
    ax.set_ylabel('Im'+r'$(\alpha)$')
    clear_output(wait=True)
    #display(f)
    #print 'nr%s (%s)' % (cost , best cost),
    return cost , grad

init_disps = np.random.normal(0, 1, 2*n_disps)#random numbers 2*226
init_disps[0] = init_disps[n_disps] = 0#putting zeros, corresponding to 0,0j
ret = minimize(wrap_cost , init_disps , method='L-BFGS-B', jac=True , options=dict(ftol=1e-6))

print(ret.message)

new_disps = ret.x[:n_disps] + 1j*ret.x[n_disps:]
print(new_disps)
# new_disps = np.concatenate(([0], new_disps ))