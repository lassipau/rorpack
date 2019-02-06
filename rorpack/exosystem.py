'''Functionality for constructing and simulating exosystems. [Not used in the current version].

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from scipy.integrate import solve_ivp
from .utilities import vectorize


class ExoSystem:
    '''
    A base class for an exosystem generating the reference signal yref(t) and the disturbance signal wdist(t).
    More precisely, 

     :math:`\\begin{align} \dot{v}(t) &= Sv(t), \quad v(0) = v_0 \in W \\\\ w_{dis}t(t) &= Ev(t) \\\\ y_{ref} &= -Fv(t)  \end{align}` 

    on a finite-dimensional space :math:`W = \mathbb{C}^r`. 
    '''

    def __init__(self, S, F, E=None):
        '''
        Parameters
        ----------
        S : (M, M) array_like
            The matrix S of the exosystem.
        F : (N, M) array_like
            The matrix F of the exosystem.
        E : (N, M) array_like, optional
            The matrix E of the exosystem. The default value of E is the
            identity matrix.
        '''
        self.S = S
        self.F = F
        if E is None:
            self.E = np.eye(S.shape[0])
        else:
            self.E = E

    def is_consistent(self):
        '''
        Checks if the dimensions of the matrices in the exosystem are consistent.

        Returns
        -------
        consistent : bool
            True if the dimensions are consistent, False otherwise.
        '''
        return self.S.shape[1] == self.E.shape[1] and self.S.shape[1] == self.F.shape[1]

    def solve(self, t_start, t_end, v0, **options):
        '''
        Simulates the exosystem to define the functions :math:`y_{ref}` and
        :math:`w_{dist}`.

        Parameters
        ----------
        t_start : float
            The starting time for the solution.
        t_end : float
            The end time for the solution.
        v0 : (,N) array_like
            The initial condition for the system at start time.
        options :
            Options passed to the differential equation solver 'solve_ivp' of SciPy.

        Returns
        -------
        yref : function
            The reference signal.
        wdist : function
            The disturbance signal.
        '''
        f = lambda t, v: np.dot(self.S, v)
        sol = solve_ivp(f, (t_start, t_end), v0, vectorized=True,
                        method='BDF', dense_output=True, **options)
        yref = vectorize(lambda t: np.atleast_2d(-np.dot(self.F, sol.sol(t))).T)
        wdist = vectorize(lambda t: np.atleast_2d(np.dot(self.E, sol.sol(t))).T)
        return yref, wdist
