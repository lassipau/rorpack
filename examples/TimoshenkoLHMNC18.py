'''
Robust output tracking of a passive Timoshenko beam model with distributed inputs and outputs.

Originally written for the conference presentation at Lagrangian and Hamiltonian Methods in
Nonlinear Control (LHMNC) in Valparaiso, Chile in 2018.

The simulation is associated to the conference paper by Paunonen, Le Gorrec, and Ramirez at LHMNC 2018
(the conference paper does not include the simulation).

The controller is a "simple passive controller" studied in the conference paper
(also see Rebarber-Weiss 2003).

The system has collocated distributed inputs and outputs,
and in the simulation it is approximated using a Finite Differences scheme.
'''

import numpy as np
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import interp1d
from rorpack.system import LinearSystem
from rorpack.controller import *
from rorpack.closed_loop_system import ClosedLoopSystem
from rorpack.plotting import *

def construct_TimoshenkoLHMNC18(w0fun, wd0fun, phi0fun, phid0fun, N):
    h = 1/N

    bw = 2
    bphi = 2
    xis = np.linspace(0,1,N+1)

    # The simulation uses values \rho=1 and I_\rho=1 (hard-coded in operators
    # A, B, C).
    rho = 1
    I_rho = 1

    ee = np.ones((1,N))

    P012 = (1/h*spdiags(np.vstack((-ee, ee)), [-1, 0], N, N).todense())
    P021 = (1/h*spdiags(np.vstack((-ee, ee)), [0, 1], N, N).todense())

    A11 = np.bmat([[np.zeros((N, N)), P012], [P021, -bw*np.eye(N)]])
    A22 = np.bmat([[np.zeros((N, N)), P012], [P021, -bphi*np.eye(N)]])

    P114 = spdiags(np.vstack((-ee, 0*ee)), [-1, 0], N, N).todense()
    P141 = -P114.T

    ZN = np.zeros((N, N))

    A12 = np.bmat([[ZN, P114], [ZN, ZN]])
    A21 = np.bmat([[ZN, ZN], [P141, ZN]])

    A = np.bmat([[A11, A12], [A21, A22]])

    # Control and observation on [4/5 1], both on the state variable x_4(t)
    B = np.bmat([[np.zeros((3*N, 1))], [np.atleast_2d(xis[1:] >= 0.8).T]])
    C = h*B.T

    Bd = np.zeros((4*N, 1))
    D = np.zeros((1, 1))

    spgrid = xis

    # Compute the initial state based on w0fun, wd0fun, phi0, and phid0
    x0 = np.hstack((np.diff(w0fun(xis))/h - phi0fun(xis[:-1]), rho*wd0fun(xis[1:]), np.diff(phi0fun(xis))/h, I_rho*phid0fun(xis[1:])))

    return LinearSystem(A, B, C, D, Bd), spgrid, x0


# Parameters for this example.
N = 50

# Initial state of the plant (the state has four components, x_1, ... x_4)
# The initial state is based on the initial data for w(xi,t) and phi(xi,t),
# namely w0fun (initial deflection), wd0fun (initial velocity), 
# phi0fun (initial angular displacement), phid0fun (initial angular
# velocity)

w0fun = lambda x: np.cos(np.pi*x)
# wd0fun = lambda x: np.zeros(np.atleast_1d(x).shape)
wd0fun = lambda x: 5*np.sin(np.pi*x)
phi0fun = lambda x: np.zeros(np.atleast_1d(x).shape)
phid0fun = lambda x: np.zeros(np.atleast_1d(x).shape)

# Construct the Timoshenko beam model from the conference article by
# Paunonen, Le Gorrec and Ramirez at LHMNC in 2018.
sys, spgrid, x0 = construct_TimoshenkoLHMNC18(w0fun, wd0fun, phi0fun, phid0fun, N)

# Define the reference and disturbance signals
# NOTE: The system has a transmission zero at s=0, and thus the tracking 
# and rejection of signal parts with this frequency is not possible! (i.e.,
# freqsReal should not contain 0).

#yref = lambda t: np.sin(2*t) + 0.1*np.cos(6*t)
#yref = lambda t: np.sin(2*t) + 0.2*np.cos(3*t)
#yref = lambda t: np.ones(np.atleast_1d(t).shape)

#wdist = lambda t: np.ones(np.atleast_1d(t).shape)
#wdist = lambda t: np.sin(t)
#wdist = lambda t: np.zeros(np.atleast_1d(t).shape)

# Case 1:
yref = lambda t: np.sin(2*t) + 0.5*np.cos(1*t)
wdist = lambda t: np.zeros(np.atleast_1d(t).shape)

# Case 2:
# yref = lambda t: np.ones(np.atleast_1d(t).shape)
# wdist = lambda t: np.ones(np.atleast_1d(t).shape)

# Case 3:
# yref = lambda t: np.sin(2*t) + 0.1*np.cos(6*t)
# wdist = lambda t: np.sin(t)

freqsReal = np.array([1, 2])


# Construct the controller

# Simple passive robust controller, used in the original Timoshenko beam
# example in the LHMNC 2018 conference paper (simulation not included in
# the paper).

dimY = sys.C.shape[0]
epsgain = np.array([3, 7])
# epsgain = 13;
contr = PassiveRC(freqsReal, dimY, epsgain, sys)

# # Alternative controller:
# # An observer-based robust controller
# # Stabilizing state feedback and output injection operators K and L
# # These are chosen based on collocated design. 
# K21 = -0.5*sys.B.conj().T
# L = -0.5*sys.C.conj().T
# IMstabmethod = 'poleplacement'
# # IMstabmethod = 'LQR'
# IMstabmargin = 0.5

# contr = ObserverBasedRC(sys, freqsReal, K21, L, IMstabmargin, IMstabmethod)


# Construct the closed-loop system
clsys = ClosedLoopSystem(sys, contr)

# Define the initial state of the closed-loop system
# (the controller has zero initial state by default).
z0 = np.zeros(contr.G1.shape[0])
xe0 = np.concatenate((x0, z0))

# Simulate the system.
# Length of the simulation
t_begin = 0
t_end = 16
t_points = 300
tgrid = np.linspace(t_begin, t_end, t_points)
# Plot the reference signal
plt.plot(tgrid,yref(tgrid))
plt.title('The reference signal $y_{ref}(t)$')
plt.grid(True)
plt.show()

sol, output, error, control, t = clsys.simulate(xe0, tgrid, yref, wdist)
print('Simulation took %.2f seconds' % t)

# Finally, plot and animate the simulation.
plot_output(tgrid, output, yref, 'samefig', 'default') 
plot_error_norm(tgrid, error)
plot_control(tgrid, control)

# In order to visualize the behaviour of the beam, we need to compute the
# deflection profile w(xi,t) based on the state variable x_2(t) =
# \rho*\dot{w}(\xi,t) by numerical integration. 
# We use a denser grid for 't' for the numerical integration
tt_int = np.linspace(0,t_end,601)
f = lambda t, xe: np.dot(clsys.Ae, xe) + np.dot(clsys.Be, np.vstack((wdist(t), yref(t))))
sol_int = solve_ivp(f, (tt_int[0], tt_int[-1]), xe0, vectorized=True, t_eval=tt_int, method='BDF')
xe_int = sol_int.y
# In this example rho=1
rho = 1
profile_int = 1/rho*(np.dot(np.atleast_2d(w0fun(spgrid)).T, np.ones((1, len(tt_int)))) + np.vstack((np.zeros((1, len(tt_int))), cumtrapz(xe_int[N:(2*N), :], tt_int, initial=0))))
# Interpolate the data to the plotting grid
interp_fun = interp1d(tt_int, profile_int)
profile = interp_fun(tgrid)

plot_1d_surface(tgrid, spgrid, profile, colormap=cm.plasma)
animate_1d_results(spgrid, profile, tgrid)