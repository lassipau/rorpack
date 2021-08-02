'''
Heat equation on the interval [0,1] with  Neumann boundary control and
Dirichlet boundary observation. Approximation with a finite difference scheme.

Neumann boundary control and disturbance at x=0, regulated output y(t)
at x=0, Dirichlet boundary condition at x=1

The transfer function values and other parameters used in the controller construction are computed using the Chebfun package (chebfun.org/) via a Matlab interface.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from rorpack.system import LinearSystem
from rorpack.controller import *
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *
from laplacian import diffusion_op_1d
# from external import chebfun_interface as cfi
from rorpack.utilities import stability_margin

def construct_heat_1d_2(N, cfun):
    spgrid = np.linspace(0, 1, N+1)

    plt.plot(spgrid,cfun(spgrid))
    plt.title('The thermal diffusivity $c(x)$ of the material')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    h = spgrid[1]-spgrid[0]
    DiffOp, spgrid = diffusion_op_1d(spgrid, cfun, 'ND')
    A = DiffOp.todense()

    B = np.bmat([[np.atleast_2d((2*cfun(0))/h)], [np.zeros((N-1, 1))]])
    Bd = B
    C = np.bmat([[np.atleast_2d(1), np.zeros((1, N-1))]])
    D = np.zeros((1, 1))
    return LinearSystem(A, B, C, D, Bd), spgrid



# Parameters for this example.
N = 50

# The spatially varying thermal diffusivity of the material
# IMPORTANT: To use Chebfun, we also define a string 'cfunML' using Matlab syntax.
# cfun = lambda x: np.ones(np.atleast_1d(x).shape)
# cfunML = '1'
# cfun = lambda x: .1+.2*x
# cfunmL = '.1+.2*x'
# cfun = lambda x: 1-2*x*(1-2*x)
# cfunML = '1-2*x*(1-2*x)'
cfun = lambda x: 1+.5*np.cos(5/2*np.pi*x)
cfunML = '1+.5*cos(5/2*pi*x)'
# Note: Lower diffusivity is difficult for the Low-Gain and Passive controllers
# cfun = lambda x: .2-.4*x*(1-x) 
# cfunML = '.2-.4*x*(1-x)'

# Length of the simulation
t_begin = 0
t_end = 14
t_points = 300

# Construct the system.
sys, spgrid = construct_heat_1d_2(N, cfun)

# yref = lambda t: np.sin(2*t) + 0.1*np.cos(6*t)
# yref = lambda t: np.sin(2*t) + 0.2*np.cos(3*t)
# yref = lambda t: np.ones(np.atleast_1d(t).shape)

# wdist = lambda t: np.ones(np.atleast_1d(t).shape)
# wdist = lambda t: np.sin(t)
# wdist = lambda t: np.zeros(np.atleast_1d(t).shape)

# Case 1:
yref = lambda t: np.sin(2*t) + .5 * np.cos(3*t)
wdist = lambda t: np.zeros(np.atleast_1d(t).shape)
# wdist = lambda t: np.sin(6*t) - 2*np.atleast_1d(1)

# Case 2:
# yref = lambda t: np.ones(np.atleast_1d(t).shape)
# wdist = lambda t: np.ones(np.atleast_1d(t).shape)

# Case 3:
# yref = lambda t: np.sin(2*t) + 0.1*np.cos(6*t)
# wdist = lambda t: np.sin(t)

freqsReal = np.array([0, 1, 2, 3, 6])


# Construct the controller

# Low-Gain Robust Controller
# Compute P(i*w_k) using Chebfun/Matlab
# Pvals = cfi.heat_1d_2_Pvals(cfunML,freqsReal)
# # Alternative (without Matlab) Computation using the FD approximation:
# # Pvals = np.array(list(map(sys.P, 1j * freqsReal)))
# epsgainrange = np.array([0.1,0.6])
# contr = LowGainRC(sys, freqsReal, epsgainrange, Pvals)

# Passive Robust Controller
# epsgainrange = np.array([0.1,0.7])
# dim_Y = sys.C.shape[0]
# # Pvals = np.array(list(map(sys.P, 1j * freqsReal)))
# contr = PassiveRC(freqsReal, dim_Y, epsgainrange, sys)

# Observer-Based Robust Controller
# Requires stabilizing operators K21 and L
# and the transfer function values P_K(i*w_k) 
# IMstabmargin = 0.3
# IMstabmethod = 'LQR'
# # IMstabmethod = 'poleplacement'
# L = - np.ones((N,1))
# K21 = - 1/(N-1)*np.bmat([[np.atleast_2d(0.5),np.ones((1,N-2)),np.atleast_2d(0.5)]])
# # A string representing the operator K21 (in Matlab function syntax)
# K21fun = '-1'
# K21 = np.zeros((1,N))
# # Compute P_K(i*w_k) and 'CKRKvals' using Chebfun/Matlab:
# PKvals, CKRKvals = cfi.heat_1d_2_PKvals(cfunML,freqsReal,K21fun,spgrid)
# # Alternative (without Matlab) Computation using the FD approximation:
# # CKRKvals = None
# contr = ObserverBasedRC(sys, freqsReal, K21, L, IMstabmargin, IMstabmethod, CKRKvals)

# Dual Observer-Based Robust Controller
# Requires stabilizing operators K2 and L1
# and the transfer function values P_L(i*w_k) 
# General options
IMstabmargin = 0.2
IMstabmethod = 'LQR'
# IMstabmethod = 'poleplacement'
K2 = 1/(N-1)*np.bmat([[np.atleast_2d(0.5),np.ones((1,N-2)),np.atleast_2d(0.5)]])
L1 = - np.ones((N,1))
# # A string representing the functional L1 (in Matlab function syntax)
# L1fun = '-1'
# # Compute P_K(i*w_k) and 'CKRKvals' using Chebfun/Matlab:
# PLvals, RLBLvals = cfi.heat_1d_2_PLvals(cfunML,freqsReal,L1fun,spgrid)
# Alternative (without Matlab) Computation using the FD approximation:
RLBLvals = None
contr = DualObserverBasedRC(sys, freqsReal, K2, L1, IMstabmargin, IMstabmethod)


# Construct the closed-loop system 
clsys = ClosedLoopSystem(sys, contr)

# Simulate the closed-loop system.

# x0fun = lambda x: np.zeros(np.size(x))
x0fun = lambda x: 0.5 * (1 + np.cos(np.pi * (1 - x)))
# x0fun = lambda x: 3*(1-x)+x
# x0fun = lambda x: 1/2*x**2*(3-2*x)-1
# x0fun = lambda x: 1/2*x**2*(3-2*x)-0.5
# x0fun = lambda x: 1*(1-x)**2*(3-2*(1-x))-1
# x0fun = lambda x: 0.5*(1-x)**2*(3-2*(1-x))-0.5
# x0fun = lambda x: 0.25*(x**3-1.5*x**2)-0.25
# x0fun = lambda x: 0.2*x**2*(3-2*x)-0.5


dim_X = np.size(spgrid)
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

# In plotting and animating the state, fill in the homogeneous Dirichlet boundary condition at x=1
sys_state = np.vstack((sol.y[0:N],np.zeros((1,np.size(tgrid)))))
spgrid = np.concatenate((spgrid,np.atleast_1d(1)))

plot_1d_surface(tgrid, spgrid, sys_state, colormap=cm.plasma)
animate_1d_results(spgrid, sys_state, tgrid)
