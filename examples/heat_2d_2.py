'''
Robust control of a 2D heat equation with either the dual observer based
robust controller (DOBRC) or the observer based robust controller (OBRC).

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from rorpack.system import LinearSystem
from rorpack.controller import *
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *
from laplacian import laplacian_2d


def construct_heat_2d_2(N, M, cval):
    dx = 1.0 / (N - 1.0)
    dy = 1.0 / (M - 1.0)
    lapD = laplacian_2d(N, M, np.array([['N', 'N'], ['N', 'N']]))
    A = cval * -lapD.todense()/(dx*dy)

    # Neumann input from the boundary where x = 0
    B = np.zeros((N*M, 1))
    B[0:N*M:N, 0] = 1 / dy

    # Observation is the integral over the part of the boundary where y = 1
    C = np.zeros((1, N*M))
    C[0, (N-1)*M:N*M] = dx
    C[0, (N-1)*M] = C[0, N*M-1] = dx / 2

    D = np.zeros((1, 1))

    # Disturbance signal input operator.
    # Neumann input from the boundary where y = 0 and 0 <= x < 1/2.
    half_N = int(np.floor(N / 2.0))
    Bd1 = np.vstack((np.ones((half_N, 1)), np.zeros((N - half_N, 1))))
    Bd2 = 2/dx * np.hstack((Bd1, np.zeros((N, M-1))))
    Bd = np.reshape(Bd2, (-1, 1), order='F')

    return LinearSystem(A, B, C, D, Bd)


# Parameters of this example.
N = 15
M = 16
cval = 1

# Construct the system.
sys = construct_heat_2d_2(N, M, cval)

# Define the reference and disturbance signals
yref = lambda t: np.atleast_1d(np.sin(2 * t) + 0.1 * np.cos(3 * t))
wdist = lambda t: np.atleast_1d(np.sin(t))
freqsReal = np.atleast_1d([0, 1, 2, 3])


# Stabilizing operators to stabilize the single unstable eigenvalue for A+B*K_S and A+L_S*C.
# 1 for interior, 1/4 for corners and 1/2 for other boundary.
K_S = -np.ones((1, N*M))
K_S[0, 0:M] /= 2.0
K_S[0, (N-1)*M:N*M] /= 2.0
K_S[0, 0:N*M:N] /= 2.0
K_S[0, M-1:N*M:N] /= 2.0
L_S = K_S.T.conj() * 10

# Construct the controller 

# Observer-based robust controller
# Requires stabilizing operators K21 and L 
K21 = K_S
L = L_S
IMstabmargin = 0.5
IMstabmethod = 'LQR'
contr = ObserverBasedRC(sys, freqsReal, K21, L, IMstabmargin, IMstabmethod)


# Dual observer-based robust controller
# Requires stabilizing operators K2 and L1
# K2 = K_S
# L1 = L_S
# IMstabmargin = 0.5
# IMstabmethod = 'LQR'
# contr = DualObserverBasedRC(sys, freqsReal, K2, L1, IMstabmargin, IMstabmethod)

# Construct the closed-loop system
clsys = ClosedLoopSystem(sys, contr)

# Simulate the system.
t_begin = 0
t_end = 10
t_points = 300
x0fun = lambda x, y: np.cos(np.pi * (1.0 - x))
xxg, yyg = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, M))
x0 = x0fun(xxg, yyg).flatten(order='F')
# z0 is chosen to be zero by default
z0 = np.zeros(contr.G1.shape[0])
xe0 = np.concatenate((x0, z0))

tgrid = np.linspace(t_begin, t_end, t_points)
sol, output, error, control, t = clsys.simulate(xe0, tgrid, yref, wdist)
print('Simulation took %f seconds' % t)

# Finally, plot and animate the simulation.
plot_output(tgrid, output, yref, 'samefig', 'default') 
plot_error_norm(tgrid, error)
plot_control(tgrid, control)
animate_2d_results(xxg, yyg, sol.y, tgrid, colormap=cm.plasma)
