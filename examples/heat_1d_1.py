'''
Heat equation on the interval [0,1] with  Neumann boundary control and
Dirichlet boundary observation. Approximation with a finite difference scheme.

Neumann boundary control at x = 0, regulated output y(t) and a
Neumann boundary disturbance at x=1. Unstable system, stabilization
by stabilizing the only unstable eigenvalue, which is 0.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from rorpack.system import LinearSystem
from rorpack.controller import *
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *
from laplacian import laplacian_1d, diffusion_op_1d


def construct_heat_1d_1(N, cfun):
    spgrid = np.linspace(0, 1, N)

    plt.plot(spgrid,cfun(spgrid))
    plt.title('The thermal diffusivity $c(x)$ of the material')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    h = spgrid[1]-spgrid[0]
    DiffOp, spgrid = diffusion_op_1d(spgrid, cfun, 'NN')
    A = DiffOp.todense()

    B = np.bmat([[np.atleast_2d((2*cfun(0))/h)], [np.zeros((N-1, 1))]])
    Bd = np.bmat([[np.zeros((N-1, 1))], [np.atleast_2d((2*cfun(1))/h)]])
    C = np.bmat([[np.zeros((1, N-1)), np.atleast_2d(1)]])
    D = np.zeros((1, 1))
    return LinearSystem(A, B, C, D, Bd), spgrid


# Parameters for this example.
N = 51

# The spatially varying thermal diffusivity of the material
# cfun = lambda x: np.ones(np.atleast_1d(x).shape)
# cfun = lambda x: 1+x
# cfun = lambda x: 1-2*x*(1-2*x)
cfun = lambda x: 1+.5*np.cos(5/2*np.pi*x)
# cfun = lambda x: .3-.6*x*(1-x)

# Length of the simulation
t_begin = 0
t_end = 14
t_points = 300

# Construct the system.
sys, spgrid = construct_heat_1d_1(N, cfun)

# yref = lambda t: np.sin(2*t) + 0.1*np.cos(6*t)
# yref = lambda t: np.sin(2*t) + 0.2*np.cos(3*t)
# yref = lambda t: np.ones(np.atleast_1d(t).shape)

# wdist = lambda t: np.ones(np.atleast_1d(t).shape)
# wdist = lambda t: np.sin(t)
# wdist = lambda t: np.zeros(np.atleast_1d(t).shape)

# Case 1:
yref = lambda t: np.sin(2*t)  # + 2 * np.cos(2*t)
# wdist = lambda t: np.zeros(np.atleast_1d(t).shape)
wdist = lambda t: np.sin(2*t)

# Case 2:
# yref = lambda t: np.ones(np.atleast_1d(t).shape)
# wdist = lambda t: np.ones(np.atleast_1d(t).shape)

# Case 3:
# yref = lambda t: np.sin(2*t) + 0.1*np.cos(6*t)
# wdist = lambda t: np.sin(t)

freqsReal = np.array([1, 2, 3, 6])

# Construct the controller 

# Observer-Based Robust Controller
# Requires stabilizing operators K21 and L
# and the transfer function values P_K(i*w_k) 
K21 = -7 * np.bmat([[np.atleast_2d(1), np.zeros((1, N - 1))]])
L = -7 * np.bmat([[np.zeros((N-1, 1))], [np.atleast_2d(2*(N-1))]])
PKvals = np.array(list(map(lambda freq: sys.P_K(freq, K21), 1j * freqsReal)))
IMstabmargin = 0.45
# IMstabmethod = 'poleplacement'
IMstabmethod = 'LQR'
contr = ObserverBasedRC(sys, freqsReal, PKvals, K21, L, IMstabmargin, IMstabmethod)

# Dual Observer-Based Robust Controller
# Requires stabilizing operators K2 and L1
# and the transfer function values P_K(i*w_k) 
# K2 = -7 * np.bmat([[np.atleast_2d(1), np.zeros((1, N - 1))]])
# L1 = -7 * np.bmat([[np.zeros((N-1, 1))], [np.atleast_2d(2*(N-1))]])
# PLvals = np.array(list(map(lambda freq: sys.P_L(freq, L1), 1j * freqsReal)))
# IMstabmargin = 0.5
# IMstabmethod = 'poleplacement'
# # IMstabmethod = 'LQR'
# contr = DualObserverBasedRC(sys, freqsReal, PLvals, K2, L1, IMstabmargin, IMstabmethod)

# Construct the closed-loop system
clsys = ClosedLoopSystem(sys, contr)


# Simulate the system.
# x0fun = lambda x: np.zeros(np.size(x))
x0fun = lambda x: 0.5 * (1 + np.cos(np.pi * (1 - x)))
# x0fun = lambda x: 3*(1-x)+x
# x0fun = lambda x: 1/2*x**2*(3-2*x)-1
# x0fun = lambda x: 1/2*x**2*(3-2*x)-0.5
# x0fun = lambda x: 1*(1-x)**2*(3-2*(1-x))-1
# x0fun = lambda x: 0.5*(1-x)**2*(3-2*(1-x))-0.5
# x0fun = lambda x: 0.25*(x**3-1.5*x**2)-0.25
# x0fun = lambda x: 0.2*x**2*(3-2*x)-0.5

x0 = x0fun(spgrid)
# z0 is chosen to be zero by default
z0 = np.zeros(contr.G1.shape[0])
xe0 = np.concatenate((x0, z0))

tgrid = np.linspace(t_begin, t_end, t_points)
sol, output, error, control, t = clsys.simulate(xe0, tgrid, yref, wdist)

# Finally, plot and animate the simulation.
plot_output(tgrid, output, yref, 'samefig', 'default') 
plot_error_norm(tgrid, error)
plot_control(tgrid, control)
plot_1d_surface(tgrid, spgrid, sol.y, colormap=cm.plasma)
animate_1d_results(spgrid, sol.y, tgrid)
