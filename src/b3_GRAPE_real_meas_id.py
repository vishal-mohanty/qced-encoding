
import numpy as np
from qutip import *
from pathlib import Path
import matplotlib.pyplot as plt
import time
start_time = time.time()#checking how long the code takes

# Grape pulses
# grape_directory = "/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240530_Obs_meas/xquadD6_gpar/0/waves.npz"
grape_directory = r"C:\Users\phant\PycharmProjects\qced-encoding\src\playground\data.npz"

D = 6
a = destroy(10).full()
# x = (a+a.T.conj())/np.sqrt(2)
x = -1j*(a-a.T.conj())/np.sqrt(2)
# x = x@x
x = x[0:D,0:D]
f = (np.max(np.linalg.eigvals(x)))
xnorm = x/f
G_tar = xnorm

# Simulation Dimensions
cdim = 15
qdim = 2

# Mode Operators
q = destroy(qdim)
c = destroy(cdim)
qd, cd = q.dag(), c.dag()

Q = tensor(q, qeye(cdim))
C = tensor(qeye(qdim), c)
Qd, Cd = Q.dag(), C.dag()
# xobs_id = (C+Cd)/np.sqrt(2)
xobs_id = -1j*(C-Cd)/np.sqrt(2)

#obs
ug = basis(qdim, 0)
Ogpar = tensor(ug*ug.dag(),(1j*cd*c*np.pi).expm())

# Define  meanthermal excitation
nbar_cav = 0.094
nbar_qb = 0.0  # we using pre-selection!

# Hamiltonian Parameters in GHz
chi = 0.177e-3
Kerr = 3e-6
#alpha = 175.62e-3

# Coherences in ns
# T1 = np.inf#103e3
# T2 = np.inf#13e3
# Tphi = np.inf#1 / (1 / T2 - 0.5 / T1)
# cavT1 = np.inf#1.2e6
T1 = 120e3
T2 = 20e3
Tphi = 1 / (1 / T2 - 0.5 / T1)
cavT1 = 1.21e6

# Collapse Operators
c_ops = [
    np.sqrt((1 + nbar_qb) / T1) * Q,  # Qubit Relaxation
    np.sqrt(nbar_qb / T1) * Qd,  # Qubit Thermal Excitations
    np.sqrt(2 / Tphi) * Qd * Q,  # Qubit Dephasing
    np.sqrt((1 + nbar_cav) / cavT1) * C,  # Cavity Relaxation
    np.sqrt(nbar_cav / cavT1) * Cd,  # Cavity Thermal Excitations
]

# Drift Hamiltonian
H0 = (
    -2 * np.pi * chi * Cd * C * Qd * Q
    - 2 * np.pi * Kerr / 2 * Cd * Cd * C * C
    # - 2 * np.pi * alpha / 2 * Qd * Qd * Q * Q
)

def drive_amp(t, dt, drive):
    """Returns the drive amplitude for a given time"""
    drive_index = int(t // dt)

    if drive_index == len(cavQ):
        drive_index -= 1

    return drive[drive_index]

data = np.load(grape_directory, "r")

dt = data["dt"]
qubitI = data["QubitI"]
qubitQ = data["QubitQ"]
cavI = data["CavityI"]
cavQ = data["CavityQ"]

tlist = [dt * i for i in range(len(cavQ))]

H_drive = [
    [2 * np.pi * (Q + Qd), lambda t, *args: drive_amp(t, dt, qubitI)],
    [2j * np.pi * (Q - Qd), lambda t, *args: drive_amp(t, dt, qubitQ)],
    [2 * np.pi * (C + Cd), lambda t, *args: drive_amp(t, dt, cavI)],
    [2j * np.pi * (C - Cd), lambda t, *args: drive_amp(t, dt, cavQ)],
]

H = [H0, *H_drive]

# Initial State


initial = tensor(fock_dm(qdim, 0), thermal_dm(cdim, nbar_cav))

# Dynamics
options = Options(max_step=2, nsteps=1e6)
results = mesolve(
    H,
    initial,
    tlist,
    c_ops=c_ops,
    options=options,
)  # progress_bar= True)

rho_f = results.states[-1]

print(rho_f)



print("")
print("--- %s seconds ---" % (time.time() - start_time))