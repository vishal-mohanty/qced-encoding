from src.encoding.classify import Classify
''' DEFAULT PARAMETERS
# Hamiltonian parameters
N = 20
g = 1

# Time parameters of evolution
t_total = 12
t_steps = 100

# Coefficients of pump
coef_pc = 0.25
coef_pq = 0.25

# Wigner Range
xRange = (-2, 2)
yRange = (-1.5, 4.5)
'''

if __name__ == "__main__":
    Classify(compression=4,
             t_total=12, t_steps=100, coef_pc=.25, coef_pq=.25).score("wigner", alpha=1e-19)


