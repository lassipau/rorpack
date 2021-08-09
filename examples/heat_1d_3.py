'''
Heat equation on the interval [0,1] with  Neumann boundary control
and Dirichlet boundary observation. Approximation with a finite
difference scheme.

Neumann boundary disturbance at x=0, two distributed controls and
two distributed measurements regulated outputs. The controls act 
on the intervals 'IB1' and 'IB2' (Default 'IB1' = [0.3,0.4] and 
'IB2' = [0.6,0.7]) and the measurements are the average 
temperatures on the intervals 'IC1'  and 'IC2' (Default 
'IC1' = [0.1,0.2] and 'IC2' = [0.8,0.9]).

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from rorpack.system import LinearSystem
from rorpack.controller import *
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *
from laplacian import diffusion_op_1d

def construct_heat_1d_3(N, cfun, IB1, IB2, IC1, IC2, printcfun=True):
    spgrid = np.linspace(0, 1, N+1)

    if printcfun:
        plt.plot(spgrid,cfun(spgrid))
        plt.title('The thermal diffusivity $c(x)$ of the material')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    h = spgrid[1]-spgrid[0]
    DiffOp, spgrid = diffusion_op_1d(spgrid, cfun, 'ND')
    A = DiffOp.todense()

    B1 = 1/(IB1[1] - IB1[0])*np.logical_and(spgrid >= IB1[0], 
            spgrid <= IB1[1])
    # Floating point rounding errors avoided with np.isclose()    
    B2 = 1/(IB2[1] - IB2[0])*np.logical_and(np.logical_or(spgrid >= IB2[0],
            np.isclose(spgrid - IB2[0], 0)), np.logical_or(spgrid <= IB2[1], np.isclose(spgrid - IB2[1], 0)))
    B = np.stack((B1, B2), axis=1)
    C1 = h/(IC1[1] - IC1[0])*np.logical_and(spgrid >= IC1[0],
            spgrid <= IC1[1])
    C2 = h/(IC2[1] - IC2[0])*np.logical_and(spgrid >= IC2[0],
            spgrid <= IC2[1])
    C = np.stack((C1, C2))
    D = np.zeros((2, 2))
    Bd = np.bmat([[np.atleast_2d((2*cfun(0))/h)], [np.zeros((N-1, 1))]])

    return LinearSystem(A, B, C, D, Bd, np.zeros((2,1))), spgrid


# Parameters for this example.
N = 100

# The spatially varying thermal diffusivity of the material
# cfun = lambda x: np.ones(np.atleast_1d(x).shape)
# cfun = lambda x: 1+x
# cfun = lambda x: 1-2*x*(1-2*x)
cfun = lambda x: 1+.5*np.cos(5/2*np.pi*x)
# Note: Lower diffusivity is difficult for the Low-Gain and Passive controllers
# cfun = lambda x: .2-.4*x*(1-x)



# Regions of inputs and outputs
IB1 = np.array([0.3, 0.4])
IB2 = np.array([0.6, 0.7])
IC1 = np.array([0.1, 0.2])
IC2 = np.array([0.8, 0.9])

# Length of the simulation
t_begin = 0
t_end = 8
t_points = 300


# Construct the system.
sys, spgrid = construct_heat_1d_3(N, cfun, IB1, IB2, IC1, IC2)



# Define the reference and disturbance signals, and list the
# required frequencies in 'freqsReal'
# Case 1:
# yref = lambda t: np.vstack((np.sin(2*t), 2*np.cos(3*t)))
# wdist = lambda t: np.atleast_2d(np.sin(6*t))
# freqsReal = np.array([2, 3, 6])


# Case 2:
# yref = lambda t: np.ones((2,np.atleast_1d(t).shape[0]))
# wdist = lambda t: np.zeros(np.atleast_1d(t).shape)
yref = lambda t: np.vstack((2*np.ones(np.atleast_1d(t).shape), 2*np.cos(3*t)+np.sin(2*t)))
wdist = lambda t: np.atleast_2d(6*np.cos(t))
freqsReal = np.array([0, 1, 2, 3, 6])



# Construct the controller 

# Low-Gain robust controller
# Requires the transfer function values P(i*w_k)
# epsgainrange = np.array([0.3,0.6])
# Pvals = np.array(list(map(sys.P, 1j * freqsReal)))
# contr = LowGainRC(sys, freqsReal, epsgainrange, Pvals)

# Dual observer-based controller
# Requires stabilizing operators K2 and L1
# K2 = -sys.B.conj().T
# L1 = -10*sys.C.conj().T
# IMstabmargin = 0.5
# IMstabmethod = 'LQR'
# contr = DualObserverBasedRC(sys, freqsReal, K2, L1, IMstabmargin, IMstabmethod)


# Observer-based controller
# Requires stabilizing operators K21 and L
# K21 = -sys.B.conj().T
# L = -10*sys.C.conj().T
# IMstabmargin = 0.5
# IMstabmethod = 'LQR'
# contr = ObserverBasedRC(sys, freqsReal, K21, L, IMstabmargin, IMstabmethod)


# A Reduced Order Observer Based Robust Controller
# The construction of the controller uses a Galerkin approximation
# of the heat system:
# The Galerkin approximation used in the controller
# design is a lower dimensional numerical approximation
# of the PDE model.
Nlow = 50
sysApprox, spgrid_unused = construct_heat_1d_3(Nlow, cfun, IB1, IB2, IC1, IC2, False)

# Parameters for the stabilization step of the controller design
alpha1 = 1.5
alpha2 = 1
Q0 = np.eye(IMdim(freqsReal, sysApprox.C.shape[0])) # Size = dimension of the IM 
Q1 = np.eye(sysApprox.A.shape[0]) # Size = dim(V_N)
Q2 = np.eye(sysApprox.A.shape[0]) # Size = dim(V_N)
R1 = np.eye(sysApprox.C.shape[0]) # Size = dim(Y)
R2 = np.eye(sysApprox.B.shape[1]) # Size = dim(U)

# Size of the final reduced-order observer part of the controller
ROMorder = 3

contr = ObserverBasedROMRC(sysApprox, freqsReal, alpha1, alpha2, R1, R2, Q0, Q1, Q2, ROMorder)

# Construct the closed-loop system 
clsys = ClosedLoopSystem(sys, contr)


# Define the initial state x_0
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
print('Simulation took %f seconds' % t)

# Plot the output and the error, and animate the behaviour 
# of the controlled state.
plot_output(tgrid, output, yref, 'subplot', 'default')
# plot_output(tgrid, output, yref, 'samefig', 'default')
plot_error_norm(tgrid, error)
plot_control(tgrid, control)

# In plotting and animating the state, fill in the homogeneous Dirichlet boundary condition at x=1
sys_state = np.vstack((sol.y[0:N],np.zeros((1,np.size(tgrid)))))
spgrid = np.concatenate((spgrid,np.atleast_1d(1)))

plot_1d_surface(tgrid, spgrid, sys_state, colormap=cm.plasma)
animate_1d_results(spgrid, sys_state, tgrid)

