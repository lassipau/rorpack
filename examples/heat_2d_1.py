'''Robust control of a 2D heat equation on a rectangle.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''


import numpy as np
from rorpack.system import LinearSystem
from rorpack.exosystem import ExoSystem
from rorpack.controller import *
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *
from rorpack.utilities import fix_freqs
from laplacian import laplacian_2d


# Computers eigenfunctions of A. 
# phi_00=1 
# phi_n0=sqrt(2)*cos(pi*n*x1) for n>=1
# phi_0m=sqrt(2)*cos(pi*m*x2) for m>=1
# phi_nm=sqrt(2)*cos(pi*n*x1)*cos(pi*m*x2) for n,m>=1
def phinm(x1, x2, n, m):
    if n == 0 and m == 0:
        return np.ones(x1.shape)
    if n == 0 and m >= 1:
        return np.sqrt(2) * np.cos(np.pi * m * x2)
    if n >= 1 and m == 0:
        return np.sqrt(2) * np.cos(np.pi * n * x1)
    if n >= 1 and m >= 1:
        return 2 * np.cos(np.pi * n * x1) * np.cos(np.pi * m * x2)
    return 0


def normalize_heat_2d_data(N, data, xgrid, ygrid):
    nn = np.arange(0, N)
    mm = np.arange(0, N)
    t_end = data.shape[1]
    result = np.zeros((xgrid.shape[0], xgrid.shape[1], t_end), 'complex')

    for i in range(0, t_end):
        gat = np.reshape(data[0:N*N, i], (N, N))
        for j in range(0, N*N):
            k1 = int(j / N)
            k2 = int(j % N)
            result[:, :, i] += gat[k1, k2] * phinm(xgrid, ygrid, nn[k1], mm[k2])

    return np.reshape(result, (xgrid.shape[0]**2, t_end))


def construct_heat_2d_1(N):
    M = N
    # Construct the system using modal approximation.
    nn = np.atleast_2d(np.arange(0, N)).T
    mm = np.atleast_2d(np.arange(0, M))   
    nnmm = np.dot(nn, np.ones((1, M))) + np.dot(np.ones((N, 1)), mm)
    glnm = np.reshape(-nnmm**2 * np.pi**2, (1, -1))
    # The original, unstable system.
    A_0 = np.diag(glnm[0])

    b11 = np.atleast_2d(0.5)
    b12 = 1/np.sqrt(2)*np.ones((1, M-1))
    b13 = np.sqrt(2)*np.sin(nn[1:]*np.pi/2)/(nn[1:]*np.pi)
    b14 = np.dot(2*np.sin(nn[1:]*np.pi/2)/(nn[1:]*np.pi), np.ones((1, M-1)))
    b1 = np.bmat([[b11, b12], [b13, b14]])
 
    b21 = np.atleast_2d(0.5)
    b22 = 1/np.sqrt(2)*(-1)**mm[:, 1:]
    b23 = -np.sqrt(2)*np.sin(nn[1:]*np.pi/2)/(nn[1:]*np.pi)
    b24 = np.dot(2*np.sin(nn[1:]*np.pi/2)/(nn[1:]*np.pi), (-1)**(mm[:, 1:]+1))
    b2 = np.bmat([[b21, b22], [b23, b24]])

    B = np.hstack((b1.flatten(order='F').T, b2.flatten(order='F').T))
    C = 2 * B.conj().T
    D = np.zeros((C.shape[0], B.shape[1]))
    Bd = np.zeros((N*M, 1))
    # Exponential stabilization with negative output feedback.
    A = A_0 - np.dot(B, C)
    
    return LinearSystem(A, B, C, D, Bd)


# Parameters for this example.
N = 31
t_begin = 0
t_end = 16
t_points = 300
tgrid = np.linspace(t_begin, t_end, t_points)

# Construct the system.
sys = construct_heat_2d_1(N)

# Define the reference and disturbance signals
yref = lambda t: np.vstack((-1 * np.ones(np.atleast_1d(t).shape), np.cos(np.pi * t)))
wdist = lambda t: np.zeros((1, np.atleast_1d(t).shape[0]))
freqsReal = np.array([0, np.pi])


# Construct the controller 

# # Low-Gain Robust Controller for a stable system
# # Requires the transfer function values P(i*w_k)
# # epsgainrange = np.array((0.4,0.7))
# epsgainrange = 0.5
# Pvals = np.array(list(map(sys.P, 1j * freqsReal)))
# contr = LowGainRC(sys, freqsReal, epsgainrange, Pvals)

# Passive Robust Controller
epsgainrange = np.array([1,2.5])
dim_Y = sys.C.shape[0]
contr = PassiveRC(freqsReal, dim_Y, epsgainrange, sys)


# Construct the closed-loop system
clsys = ClosedLoopSystem(sys, contr)

# Simulate the system. Also print the computation time of the simulation.
x0 = np.zeros(N*N, 'complex')
# z0 is chosen to be zero by default
z0 = np.zeros(contr.G1.shape[0])
xe0 = np.concatenate((x0, z0))

sol, output, error, control, t = clsys.simulate(xe0, tgrid, yref, wdist)
print('Simulation took %.2f seconds' % t)

# Finally, plot the simulation.
plot_output(tgrid, output, yref, 'subplot', 'default') 
# plot_output(tgrid, output, yref, 'samefig', 'default') 
plot_error_norm(tgrid, error)
plot_control(tgrid, control)
xxg, yyg = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
animate_2d_results(xxg, yyg, normalize_heat_2d_data(N, sol.y, xxg, yyg), tgrid, colormap=cm.plasma)
