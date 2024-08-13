import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import scipy as sc
from scipy.linalg import expm
import os
# new_directory = r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20240530_Obs_meas"
# new_directory = "/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240530_Obs_meas"
os.chdir(new_directory)
from TK_basics import *

# Simulation Dimensions
cdim = 30
qdim = 2
ug = basis(qdim,0).full()
# ue = basis(qdim,1)

data_dr = np.load("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240530_Obs_meas/xquadD6_gpar/0/waves.npz", "r")
# data_dr = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20240530_Obs_meas\xquadD6_gpar\0\waves.npz", 'r')
# data_dr = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20240530_Obs_meas\pquadD6_gpar\0\waves.npz", 'r')

dt = data_dr['dt']
# the drives are already in GHz
qubitI = data_dr['QubitI']
qubitQ = data_dr['QubitQ']
cavI = data_dr['CavityI']
cavQ = data_dr['CavityQ']

# Mode Operators
q = destroy(qdim).full()
c = destroy(cdim).full()
qd, cd = q.T.conj(), c.T.conj()

Q = np.kron(q, qeye(cdim))
C = np.kron(qeye(qdim), c)
Qd, Cd = Q.T.conj(), C.T.conj()

# Hamiltonian Parameters in GHz
chi = 1.48e-3
Kerr = 3e-6
chi_prime = 0*15.76e-6
anharm = 0*175.31e-3
# alpha = 224.4e-3
# alpha *= 2*pi

# Drift Hamiltonian
H0 = -2*np.pi*chi*Cd@C@Qd@Q - 2*np.pi*Kerr/2*Cd@Cd@C@C - 2*np.pi*chi_prime/2*Cd@Cd@C@C@Qd@Q - 2*np.pi*anharm/2*Qd@Qd@Q@Q

Hc = [
        2*np.pi*(C + Cd),
        1j*2*np.pi*(C - Cd),
        2*np.pi*(Q + Qd),
        1j*2*np.pi*(Q - Qd),
        ]

Utot = np.kron(np.eye(qdim),np.eye(cdim))
for j in range(len(qubitI)):
    Udt = expm(-1j*(H0+qubitI[j]*Hc[2]+qubitQ[j]*Hc[3]+cavI[j]*Hc[0]+cavQ[j]*Hc[1]))
    Utot = Udt@Utot

D = 6
a = destroy(10).full()
x = (a+a.T.conj())/np.sqrt(2)
# x = -1j*(a-a.T.conj())/np.sqrt(2)
# x = x@x
x = x[0:D,0:D]
f = (np.max(np.linalg.eigvals(x)))
xnorm = x/f
G_tar = xnorm

P = expm(1j*cd@c*np.pi)
Utotcav = Utot[0:cdim, 0:cdim]
Gtot = Utotcav.T.conj()@P@Utotcav
G_grape = Gtot[0:D,0:D]#the truncated U from simulation

norm = np.sum(abs(G_tar)**2)
overlap = np.sum(G_tar.conj() * G_grape) / norm
F = np.abs(overlap)
print(f'Fidelity is {F}')

#checking observables for random states within D
# r0q = thermal_dm(qdim,0).full()
x_id = ((C+Cd)/np.sqrt(2))
# x_id = -1j*((C-Cd)/np.sqrt(2))
# x_id = x_id@x_id

Ntr = 200
xe_id = np.zeros(Ntr, dtype = np.float_)
xe_sim = np.zeros(Ntr, dtype = np.float_)
for j in range(Ntr):
    #qudit mixed state embedded in the cavity mode
    rd1 = np.zeros([cdim, cdim], dtype = np.complex_)
    u_rand = rand_ket(D)
    r_rand = (u_rand*u_rand.dag()).full()
    rd1[0:D,0:D] = r_rand
    
    w = 0
    #the evolution
    rho = np.kron(ug*ug.T.conj(), rd1)
    xe_id[j] = np.real(np.trace(x_id@rho))
    # xe_sim[j] = np.real(np.trace(np.kron(ug*ug.T.conj(),P)@Utot@rho@Utot.T.conj()))*np.real(f)
    rhot = Utot@rho@Utot.T.conj()
    pg = np.real( np.trace(np.kron(ug*ug.T.conj(),np.eye(cdim))@rhot) )
    rho_collapse = np.kron(ug*ug.T.conj(),np.eye(cdim))@rhot@np.kron(ug*ug.T.conj(),np.eye(cdim))/pg
    ppar = np.real( np.trace(np.kron(np.eye(qdim),P)@rho_collapse) )
    xe_sim[j] = pg*ppar*np.real(f)

# ind = np.argsort(xe_id)
# plt.plot(range(Ntr),xe_id[ind],'sk', markersize = 10)
# plt.plot(range(Ntr),xe_sim[ind],'or')
plt.plot(xe_id, xe_sim,'ok')
plt.plot(xe_id, xe_id,'r-')
plt.xlabel('x_id')
plt.ylabel('x_sim')

plt.show()
