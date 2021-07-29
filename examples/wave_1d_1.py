'''Robust output output tracking for a 1D wave equation with boundary control and observation and possible boundary disturbance. Neumann boundary control at :math:`x=1`, velocity observation at :math:`x=0` and Neumann boundary disturbance at :math:`x=0` (non-collocated with the control input). The controller design uses an additional measured output :math:`y_m(t)` in prestabilizing the second order eigenvalue at zero.

The simulation considers output tracking of the velocity :math:`w_t(0,t)` to arbitrary 2-periodic reference signals. This is achieved by including frequencies of the form :math:`k\\pi` for :math:`k=1..q` in the internal model. The reference signal is defined by defining its profile over one period in the variable 'yref1per'. Note that it is not necessary to form the Fourier series expansion of the reference signal.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
import scipy.integrate as integrate
from rorpack.system import LinearSystem
from rorpack.controller import ObserverBasedRC
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *

import matplotlib as mpl


def phin(x, n):
    t1 = np.dot(np.atleast_2d(n == 0).T, np.ones(np.atleast_2d(x).shape))
    t2_1 = np.cos(np.dot(np.atleast_2d(n).T, np.pi * np.atleast_2d(x)))
    t2 = np.dot(np.diag(np.atleast_1d(n > 0)), np.sqrt(2) * t2_1)
    return t1 + t2


def get_wave_1d_state(data, xgrid):
    N = data.shape[0]/2
    phinvals = phin(xgrid, np.arange(N))
    ww = np.dot(data[0::2, :].T, phinvals)
    wwd = np.dot(data[1::2, :].T, phinvals)
    return ww.T, wwd.T


def construct_wave_1d_1(N, w0fun, wd0fun):
    A = np.zeros((2*N, 2*N))
    B = np.zeros((2*N, 1))
    Bd = np.zeros((2*N, 1))
    C = np.zeros((1, 2*N))
    D = np.zeros((1, 1))
    Dd = np.zeros((1, 1))
    Kinf = np.zeros((1, 2*N))
    Linf = np.zeros((2*N, 1))
    x0 = np.zeros(2*N)

    for n in range(0, N):
        indran = slice(2*n, 2*n+2)
        A[indran, indran] = np.array([[0, 1], [-n**2 * np.pi**2, 0]])
        B[indran] = np.array([[0], [phin(1, n)]])
        Bd[indran] = np.array([[0], [phin(0, n)]])
        C[0, indran] = np.array([[0, phin(0, n)]])
        Kinf[0, indran] = np.array([[0, phin(1, n)]])
        Linf[indran] = np.array([[0], [phin(0, n)]])
        x0[indran] = np.array([integrate.quad(lambda x: w0fun(x) * phin(x, n).conj(), 0, 1)[0],
                               integrate.quad(lambda x: wd0fun(x) * phin(x, n).conj(), 0, 1)[0]])

    # Additional measured output which is used in the pre-stabilization
    Cm = np.bmat([[np.array([[1, 0]]), np.zeros((1, 2*N-2))]])
    Dm = np.zeros((1, 1))
    Dmd = np.zeros((1, 1))
    # Pre-stabilization gain parameter kappa_m
    kappa_m = 0.3

    # Stabilization gains kappa (state feedback) and ell (output injection)
    kappa = 0.9
    ell = 0.8
    K_S = -kappa * Kinf
    L = -ell * Linf

    return LinearSystem(A - kappa_m * np.dot(B, Cm), B, C, D, Bd, Dd, Cm, Dm, Dmd), x0, K_S, L


# Parameters for this example.
N = 50
# Construct the system and define the initial state. The construtor routine also returns operators K_S and L_S that stabilize A+B*K_S and A+L_S*C
# ('w0' = initial profile, 'wd0' = initial velocity)
w0fun = lambda x: np.zeros(np.size(x))
# w0fun = lambda x: 1 + np.cos(3 * np.pi * x) + np.cos(6*x)
wd0fun = lambda x: np.zeros(np.size(x))
sys, x0, K_S, L_S = construct_wave_1d_1(N, w0fun, wd0fun)

# Plot the spectra of the stabilized operators A+B*K_S and A+L_S*C
# plot_spectrum(sys.A + np.dot(sys.B, K_S), [-1.5, 0.1])
# plot_spectrum(sys.A + np.dot(L_S, sys.C), [-1.5, 0.1]) 


# Define the reference and disturbance signals:
# Tracking of an arbitrary (nonsmooth) 2-periodic signal (no disturbance input)
wdist = lambda t: np.zeros(np.atleast_1d(t).shape)

# Begin by defining the function on a single period 0<t<2
# A nonsmooth triangle signal
yref1per = lambda t: (2*t-1)*(t>=0)*(t<=1)+(3-2*t)*(t>1)*(t<2) 
# Semicircles
# yref1per = lambda t: np.sqrt(1-np.square(t-1))
# Alternating semicircles
# yref1per = lambda t: np.sqrt(np.abs(1/4-np.square(t-1/2)))*(t>=0)*(t<1)-np.sqrt(np.abs(1/4-np.square(t-3/2)))*(t>=1)*(t<2)
# Bump and constant
# yref1per = lambda t: np.sqrt(np.abs(1/4-np.square(t-1)))*(t>=1/2)*(t<3/2)
# yref1per = lambda t: np.sqrt(np.abs(1/4-np.square(t-1/2)))*(t>=0)*(t<1)

# The constant part of the signal cannot be tracked due to the second order zero of the plant at zero. We therefore normalize yref(t) to have average zero on [0,2]
yr_ave = integrate.quad(yref1per,0,2.0)
yref = lambda t: yref1per(np.fmod(t,2))-yr_ave[0]/2

# Include frequencies pi*k that are required in tracking 2-periodic signals
freqsReal = np.pi*np.arange(1,30,1)

# Plot the reference signal
tt = np.linspace(0,10,501)
plt.plot(tt,yref(tt))
plt.title('The reference signal $y_{ref}(t)$')
plt.show()

# Debug: End execution here (for testing different reference signals)
# raise Exception('End')

# Construct the controller 

# Observer-Based Robust Controller
# Requires stabilizing operators K21 and L
K21 = K_S
L = L_S
IMstabmargin = 0.5
IMstabmethod = 'LQR'
contr = ObserverBasedRC(sys, freqsReal, K21, L, IMstabmargin, IMstabmethod)


# Construct the closed-loop system.
clsys = ClosedLoopSystem(sys, contr)

# Simulate the system. Initial state x0 is chosen earlier.
# z0 is chosen to be zero by default
z0 = np.zeros(contr.G1.shape[0])
xe0 = np.concatenate((x0, z0))

t_begin = 0
t_end = 10
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
