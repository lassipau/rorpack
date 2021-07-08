'''Robust output output tracking for a 1D wave equation with distrubuted control and observation and possible distributed disturbance input. Dirichlet boundary conditions at :math:`\\xi=0` and :math:`\\xi=1`. Since the control and observation are distributed, the system is only strongly (and polynomially) stabilizable. Because of this, the low-gain controller cannot be used, and the observer-based controller designs do not guarantee closed-loop stability. However, since the system is passive, the Passive Robust Controller can be used in achieving robust output regulation.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
import scipy.integrate as integrate
from rorpack.system import LinearSystem
from rorpack.controller import PassiveRC
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *
from rorpack.utilities import stability_margin

import matplotlib as mpl


def phin(x, n):
    tmp = np.sin(np.dot(np.atleast_2d(n).T, np.pi * np.atleast_2d(x)))
    return np.dot(np.diag(np.atleast_1d(n > 0)), np.sqrt(2) * tmp)


def get_wave_1d_state(data, xgrid):
    N = data.shape[0]/2
    phinvals = phin(xgrid, np.arange(1,N+1))
    ww = np.dot(data[0::2, :].T, phinvals)
    wwd = np.dot(data[1::2, :].T, phinvals)
    return ww.T, wwd.T


def construct_wave_1d_1(N, Bfun, Bdfun, w0fun, wd0fun):
    A = np.zeros((2*N, 2*N))
    B = np.zeros((2*N, 1))
    Bd = np.zeros((2*N, 1))
    C = np.zeros((1, 2*N))
    D = np.zeros((1, 1))
    Dd = np.zeros((1, 1))
    # Kinf = np.zeros((1, 2*N))
    # Linf = np.zeros((2*N, 1))
    x0 = np.zeros(2*N)

    for n in range(1, N+1):
        indran = slice(2*(n-1), 2*(n-1)+2)
        A[indran, indran] = np.array([[0, 1], [-n**2 * np.pi**2, 0]])
        B[indran] = np.vstack(([0], integrate.quad(lambda x: Bfun(x) * phin(x, n).conj(), 0, 1)[0]))
        # Disturbance input
        Bd[indran] = np.vstack(([0], integrate.quad(lambda x: Bdfun(x) * phin(x, n).conj(), 0, 1)[0]))
        C = B.T
        x0[indran] = np.array([integrate.quad(lambda x: w0fun(x) * phin(x, n).conj(), 0, 1)[0],
                               integrate.quad(lambda x: wd0fun(x) * phin(x, n).conj(), 0, 1)[0]])

    return LinearSystem(A, B, C, D, Bd, Dd), x0

# Parameters for this example.
N = 40
# Construct the system and define the initial state. 
# Input and disturbance input profiles
Bfun = lambda x: 10 * (1 - x)
Bdfun = lambda x: 5 * x * (1 - x)
# ('w0' = initial profile, 'wd0' = initial velocity)
w0fun = lambda x: np.zeros(np.size(x))
# w0fun = lambda x: 1 + np.cos(3 * np.pi * x) + np.cos(6*x)
w0fun = lambda x: x * (x - 1) * (2 - 5 * x)
wd0fun = lambda x: np.zeros(np.size(x))
sys, x0 = construct_wave_1d_1(N, Bfun, Bdfun, w0fun, wd0fun)

# Stabilizing output feedback gain
kappa_S = 1.0

# # Analyse the properties of the stabilized system
# stab_sys = sys.output_feedback(-np.atleast_2d(kappa_S))
# plot_spectrum(stab_sys.A, [-1.5, 0.1])
# print(stability_margin(stab_sys.A))

# Define the reference and disturbance signals:
# Note that the system has an invariant zero at s=0, and therefore the
# regulation of constant signals is impossible
# Case 1:
# yref = lambda t: np.sin(np.pi*t) + 0.25*np.cos(2*np.pi*t) 
# wdist = lambda t: np.sin(6*np.pi*t)
# freqsReal = np.array([np.pi, 2*np.pi, 6*np.pi])
# Case 2:
yref = lambda t: np.sin(np.pi*t) + 0.25*np.cos(2*np.pi*t) 
wdist = lambda t: np.zeros(np.atleast_1d(t).shape)
freqsReal = np.array([np.pi, 2*np.pi])


# # Alternative: Tracking of general 2-periodic reference signals
# # Tracking of an arbitrary (nonsmooth) 2-periodic signal (no disturbance input)
# wdist = lambda t: np.zeros(np.atleast_1d(t).shape)
# # 
# # Begin by defining the function on a single period 0<t<2
# # A nonsmooth triangle signal
# # yref1per = lambda t: (2*t-1)*(t>=0)*(t<=1)+(3-2*t)*(t>1)*(t<2) 
# # Semicircles
# # yref1per = lambda t: np.sqrt(1-np.square(t-1))
# # Alternating semicircles
# yref1per = lambda t: np.sqrt(np.abs(1/4-np.square(t-1/2)))*(t>=0)*(t<1)-np.sqrt(np.abs(1/4-np.square(t-3/2)))*(t>=1)*(t<2)
# # Bump and constant
# # yref1per = lambda t: np.sqrt(np.abs(1/4-np.square(t-1)))*(t>=1/2)*(t<3/2)
# # yref1per = lambda t: np.sqrt(np.abs(1/4-np.square(t-1/2)))*(t>=0)*(t<1)
# # 
# # The constant part of the signal cannot be tracked due to the second order zero of the plant at zero. We therefore normalize yref(t) to have average zero on [0,2]
# yr_ave = integrate.quad(yref1per,0,2.0)
# yref = lambda t: yref1per(np.fmod(t,2))-yr_ave[0]/2
# # Include frequencies pi*k that are required in tracking 2-periodic signals
# freqsReal = np.pi*np.arange(1,10,1)
# 
# # Plot the reference signal
# tt = np.linspace(0,10,501)
# plt.plot(tt,yref(tt))
# plt.title('The reference signal $y_{ref}(t)$')
# plt.show()
#
# Debug: End execution here (for testing different reference signals)
# raise Exception('End')

# Construct the controller 
# Passive Robust Controller
epsgainrange = np.array([0.01,0.3])
epsgainrange = 0.3
dim_Y = sys.C.shape[0]
# Pvals = np.array(list(map(sys.P, 1j * freqsReal)))
contr = PassiveRC(freqsReal, dim_Y, epsgainrange, sys, np.atleast_2d(-kappa_S))


# Construct the closed-loop system.
clsys = ClosedLoopSystem(sys, contr)

# Plot the spectrum of the closed-loop system
# plot_spectrum(clsys.Ae, [-.5, 0.1])

# Simulate the system. Initial state x0 is chosen earlier.
# z0 is chosen to be zero by default
z0 = np.zeros(contr.G1.shape[0])
xe0 = np.concatenate((x0, z0))

t_begin = 0
t_end = 24
Nt = 501
tgrid = np.linspace(t_begin, t_end, Nt)
sol, output, error, control, t = clsys.simulate(xe0, tgrid, yref, wdist)
print('Simulation took %f seconds' % t)

# Finally, plot and animate the simulation.
plot_output(tgrid, output, yref, 'samefig', 'default') 
plot_error_norm(tgrid, error)
plot_control(tgrid, control)
xgrid = np.linspace(0, 1, 100)
# Compute the simulated wave profile and velocity profile 
ww, wwd = get_wave_1d_state(sol.y[0:2*N, :], xgrid)
plot_1d_surface(tgrid, xgrid, ww)
# Animate simulated state
animate_1d_results(xgrid, ww, tgrid)
# Animate simulated velocity
# animate_1d_results(xgrid, wwd, tgrid)
