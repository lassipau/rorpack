'''
Functionality for constructing the closed-loop system of a linear system and a dynamic error feedback controller and simulation. The inputs of the closed-loop system are the reference signal :math:`y_{ref}(t)` and the disturbance input wdist(t), and its output is the regulation error :math:`e(t)=y(t)-y_{ref}(t)`.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import time
import numpy as np
from scipy.integrate import solve_ivp
from .controller import *
from .system import *
from .utilities import *


class ClosedLoopSystem:
    '''
    Construct the closed-loop system for analysis and simulation.
    '''

    def __init__(self, sys, contr, quiet=False):
        '''
        Parameters
        ----------
        sys : LinearSystem
            The controlled system.
        contr : Controller
            The robust dynamic error feedback controller.
        quiet : bool, optional
            If False, function prints the closed-loop stability margin 
            and stiffness ratio.

        Raises
        ------
        stabexception : Exception
            Raised if the constructed closed-loop system is not stable. This
            means the robust controller construction has failed for some
            reason. In robust output regulation, simulation of an unstable
            closed-loop system is never relevant.
        '''

        temp = np.dot(sys.D, contr.Dc)
        res = np.linalg.inv(np.eye(temp.shape[0], temp.shape[1]) - temp)
        D0 = np.dot(res, sys.D)
        Dd0 = np.dot(res, sys.Dd)
        Bd0 = sys.Bd + np.dot(sys.B, np.dot(contr.Dc, Dd0))
        C0 = np.dot(res, sys.C)
        temp = np.dot(contr.Dc, D0)
        B0 = np.dot(sys.B, np.eye(temp.shape[0], temp.shape[1]) + temp)
        A0 = sys.A + np.dot(sys.B, np.dot(contr.Dc, C0))

        self.Ae = np.bmat([[A0, np.dot(B0, contr.K)],
                      [np.dot(contr.G2, C0), contr.G1 + np.dot(contr.G2, np.dot(D0, contr.K))]])
        self.Be = np.bmat([[Bd0, -np.dot(B0, contr.Dc)],
                      [np.dot(contr.G2, Dd0), -np.dot(contr.G2, res)]])
        self.Ce = np.bmat([[C0, np.dot(D0, contr.K)]])
        self.De = np.hstack((Dd0, -res))

        # The output operator :math:`C_K:=[0,K]:X\times Z\rightarrow U`
        # for computing the control signal u(t) from the closed-loop
        # system state :math:`x_e(t)`.
        self.CK = np.hstack((np.zeros((sys.B.shape[1],sys.A.shape[1])),contr.K))

        if not self.is_consistent():
            raise Exception('Closed-loop system parameters have incompatible dimensions.')

        stabmargin, ratio = stability_margin_and_stiffness_ratio(self.Ae)
        if ratio is None:
            raise Exception('Closed loop system not stable, robust controller design has failed!')
        # Print the stability margin and stiffness ratio.
        if not quiet:
            print('Stability margin: %.4f' % stabmargin)
            print('Stiffness ratio: %.1f' % ratio)

            
    def is_consistent(self):
        '''
        Verifies that the dimensions of the matrices of the closed-loop 
        system are consistent.
        
        Returns
        -------
        consistent : bool
        True if the dimensions are consistent, False otherwise.
        '''
        return self.Ae.shape[0] == self.Ae.shape[1] and \
        self.Ae.shape[0] == self.Be.shape[0] and \
        self.Ce.shape[0] == self.De.shape[0] and \
        self.Be.shape[1] == self.De.shape[1] 

    def is_stable(self):
        '''
        Checks the internal stability of the closed-loop system.

        Returns
        -------
        stable : bool
        True if the closed-loop system matrix :math:`A_e` is Hurwitz, False otherwise.
        '''
        return stability_margin(self.Ae) > 0.0

    def simulate(self, xe0, tgrid, yref, wdist, **options):
        '''
        Simulates the closed loop system with a given reference signal yref(t) and a disturbance signal wdist(t).

        Parameters
        ----------
        xe0 : (, N) array_like
            The initial state x_{e0}=(x_0,z_0)^T of the closed-loop system.
        tgrid : (, M) array_like
            The time grid for the simulation.
        yref : callable
            The reference signal to be tracked. The value yref(t) should be either a scalar or a column vector of length p. For an 1D array input t of length N, the output yref(t) should have dimensions p x N.
        wdist : callable
            The disturbance signal to be rejected. The value wdist(t) should be either a scalar or a column vector of length m_d. For an 1D array input t of length N, the output wdist(t) should have dimensions m_d x N.
        options
            Options passed on to the ODE-solver in the simulation. For more
            details see scipy.integrate.solve_ivp.

        Returns
        -------
        sol
            The solution structure of scipy.integrate.solve_ivp.
        output : (N1, M1) array_like
            The output of the simulated closed-loop system evaluated 
            at the points in tgrid.
        error : (N2, M2) array_like
            The regulation error regulation error e(t)=y(t)-yref(t)        
            evaluated at the points in tgrid.
        control : (N1, M1) array_like
            The control input of the simulated system at the points in tgrid.
        t : float
            The computation time of the simulation.
        '''

        t1 = time.time()
        f = lambda t, xe: np.dot(self.Ae, xe) + np.dot(self.Be, np.vstack((wdist(t), yref(t))))
        sol = solve_ivp(f, (tgrid[0], tgrid[-1]),
                        xe0, vectorized=True, t_eval=tgrid,
                        method='BDF', **options)
        xe = sol.y
        error = np.dot(np.array(self.Ce), xe) + np.dot(np.array(self.De), np.vstack((wdist(tgrid), yref(tgrid))))
        output = error + yref(tgrid)
        control = np.dot(np.array(self.Ce), xe)
        t2 = time.time()
        return sol, output, error, control, t2 - t1
