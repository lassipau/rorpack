'''Functionality for construction and analysis of linear control systems of the purposes of internal model based controller design.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from .utilities import *


class LinearSystem:
    '''
    A general linear control system of the form

     :math:`\\begin{align} 
     \dot{x}(t) &= A x(t) + Bu(t) + B_d w_{dist}(t), \quad x(0) = x_0 \in X \\\\
     y(t) &= C x(t) + Du(t) + D_d w_{dist}(t)  \\\\
     y_m(t) &= C_m x(t) + D_m u(t) + D_{md}w_{dist}(t)
     \end{align}` 

    on a Hilbert or Banach space :math:`X`. 
    In the system:
    - x(t) is the state of the system
    - u(t) is the control or main input
    - y(t) is the main output 
    - w_{dist}(t) is the (possible) disturbance input
    - y_m(t) is the (possible) additional measured output, used in the controller design but not considered in the output tracking problem.

    If Bd is not given (or defined to be None), the disturbance input and 
    feedthrough operators (Bd,Dd) are defined to be zero matrices and
    wdist(t) is assumed to be scalar-valued.
    If Cm is not given (or defined to be None), the additional measurement
    operators (Cm,Dm,Dmd) are defined to be zero matrices with one row. 
    '''

    def __init__(self, A, B, C, D, Bd=None, Dd=None, Cm=None, Dm=None, Dmd=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # Define disturbance inputs, if neither Bd or Dd is given, it is assumed that the disturbance input wdist(t) is scalar-valued.
        if Bd is None:
            Bd = np.zeros((A.shape[0], 1))
            Dd = np.zeros((C.shape[0], 1))
        elif Dd is None:
            Dd = np.zeros((C.shape[0], Bd.shape[1]))
        self.Bd = Bd
        self.Dd = Dd

        if Cm is None:
            Cm = np.zeros((1, A.shape[1]))
            Dm = np.zeros((1, B.shape[1]))
            Dmd = np.zeros((1, Bd.shape[1]))
        else: 
            if Dm is None:
                Dm = np.zeros((Cm.shape[0], B.shape[1]))
            if Dmd is None:
                Dmd = np.zeros((Cm.shape[0], Bd.shape[1]))

        self.Cm = Cm
        self.Dm = Dm
        self.Dmd = Dmd

        if not self.is_consistent():
            raise Exception('System parameters have incompatible dimensions.')

    def is_consistent(self):
        '''
        Verifies that the dimensions of the matrices of the system 
        are consistent.

        Returns
        -------
        consistent : bool
            True if the dimensions are consistent, False otherwise.
        '''
        return self.A.shape[0] == self.A.shape[1] and \
            self.A.shape[0] == self.B.shape[0] and \
            self.C.shape[0] == self.D.shape[0] and \
            self.B.shape[1] == self.D.shape[1] and \
            self.A.shape[1] == self.C.shape[1] and \
            self.A.shape[0] == self.Bd.shape[0] and \
            self.C.shape[0] == self.Dd.shape[0] and \
            self.Bd.shape[1] == self.Dd.shape[1] and \
            self.Cm.shape[0] == self.Dm.shape[0] and \
            self.B.shape[1] == self.Dm.shape[1] and \
            self.A.shape[1] == self.Cm.shape[1] and \
            self.Cm.shape[0] == self.Dmd.shape[0] and \
            self.Bd.shape[1] == self.Dmd.shape[1] 

    def is_stable(self):
        '''
        Checks the internal stability of the system.

        Returns
        -------
        stable : bool
        True if the system matrix :math:`A` is Hurwitz, False otherwise.
        '''
        return stability_margin(self.A) > 0.0


    def P(self, s):
        '''
        Computes the transfer function :math:`P` for a particular 
        complex number 's'.

        Parameters
        ----------
        s : float
            The given complex number s.

        Returns
        -------
        P : (p, m) array_like
            The transfer function.
        '''
        res = np.linalg.solve(s * np.eye(self.A.shape[0]) - self.A, self.B)
        return np.dot(self.C, res) + self.D

    def P_L(self, s, L):
        '''
        Computes the (approximate) value of the closed-loop transfer function
        :math:`P_L(s)=CR(s,A+LC)(B+LD)+D` for a given complex
        number 's' and an output injection operator :math:`L`.

        Parameters
        ----------
        s : float
            The given complex number s.

        L : array_like
            The output injection operator :math:`L`.

        Returns
        -------
        P_L : (p, m) array_like
            The value of the closed-loop transfer function at s.
        '''
        res = np.linalg.solve(s * np.eye(self.A.shape[0]) - self.A - np.dot(L, self.C), self.B + np.dot(L, self.D))
        return np.dot(self.C, res) + self.D

    def RLBL(self, s, L):
        '''
        Computes (an approximation) of :math:`R(s,A+LC)(B+LD)` for a
        given complex number 's' and an output injection operator :math:`L`.

        Parameters
        ----------
        s : float
            The given complex number 's'.

        L : array_like
            The output injection operator :math:`L`.

        Returns
        -------
        RLBL : (n, m) array_like
            The value of the function at 's'.
        '''
        return np.linalg.solve(s * np.eye(self.A.shape[0]) - self.A - np.dot(L, self.C), self.B + np.dot(L, self.D))

    def P_K(self, s, K):
        '''
        Computes the (approximate) value of the closed-loop transfer function
        :math:`P_K(s)=(C+DK)R(s,A+BK)B+D` for a particular
        complex number 's' and a given state feedback operator :math:`K`.

        Parameters
        ----------
        s : float
            The given complex number 's'.

        K : array_like
            The state feedback operator :math:`K`.

        Returns
        -------
        P_K : (p, m) array_like
            The value of the closed-loop transfer function at 's'.
        '''
        res = np.linalg.solve(s * np.eye(self.A.shape[0]) - self.A - np.dot(self.B, K), self.B)
        return np.dot(self.C + np.dot(self.D, K), res) + self.D

    def CKRK(self, s, K):
        '''
        Computes (an approximation) of :math:`(C+DK)R(s,A+BK)` for a
        given complex number 's' and state feedback operator :math:`K`.

        Parameters
        ----------
        s : float
            The given complex number 's'.

        K : array_like
            The state feedback operator :math:`K`.

        Returns
        -------
        CKRK : (p, n) array_like
            The value of the function at 's'.
        '''
        res = np.linalg.solve(s * np.eye(self.A.shape[0]) - self.A - np.dot(self.B, K), np.eye(self.A.shape[0]))
        return np.dot(self.C + np.dot(self.D, K), res) 

    # [Unused]: Transfer function of a boundary control system.
    def P2(self, s):
        '''
        Computes the value of the transfer function :math:`P_2` of a 
        boundary control system for a given complex number 's'.

        Parameters
        ----------
        s : float
            The given complex number 's'.

        Returns
        -------
        P2 : (N, M) array_like
            The value of the transfer function at 's'.
        '''
        res = np.linalg.solve(s * np.eye(self.A.shape[0]) - self.A, -s * self.B)
        return np.dot(self.C, res) + self.D


    def output_feedback(self, K):
        '''
        Computes the linear system 
        (A+BK(I+DK)^(-1)C,B(I+KD)^(-1),(I+DK)^(-1)C,(I+DK)^(-1)D) 
        obtained from (A,B,C,D) with output feedback u(t)=Ky(t)

        Parameters
        ----------
        K : (m,p) array_like
            The output feedback operator.

        Returns
        -------
        CL_sys : LinearSystem
            The closed-loop system after the output feedback.
        '''
        dim_U = self.D.shape[1]
        dim_Y = self.D.shape[0]

        Q = np.linalg.inv(np.eye(dim_Y) - np.dot(self.D, K))
        Q2 = np.eye(dim_U) + np.dot(K, np.dot(Q,self.D))
        Bfb = np.dot(self.B, Q2)
        Cfb = np.dot(Q, self.C)
        Dfb = np.dot(Q, self.D)
        Afb = self.A + np.dot(self.B, np.dot(K, Cfb))
        return LinearSystem(Afb, Bfb, Cfb, Dfb)
