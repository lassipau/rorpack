'''Functionality for constructing different internal model based controllers for robust output regulation of linear systems.

The list of included controllers (see the documentation for more detailed information):

1. LowGainRC - 'Minimal' Low-Gain Robust Controller for stable systems.

2. ObserverBasedRC - Observer-Based Robust Controller for possibly unstable systems.

3. DualObserverBasedRC - Dual Observer-Based Robust Controller for possibly unstable systems.

4. PassiveRC - 'Minimal' Passive Robust Controller for stable passive systems.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''


import numbers
import time
import numpy as np
import scipy as sp
import control
from scipy.signal import place_poles
from .utilities import *


class Controller:
    '''
    The base class for a dynamic error feedback controller.
    '''

    def __init__(self, G1, G2, K, Dc=None, epsilon=0.0):
        '''
        Parameters
        ----------
        G1 : (dim_Z, dim_Z) array_like
            The matrix G1 of the controller.
        G2 : (dim_Z, dim_Y) array_like
            The matrix G2 of the controller.
        K : (dim_U, dim_Z) array_like
            The matrix K of the controller.
        Dc : (dim_Y, dim_U), array_like, optional
            The feedthrough operator of the controller, not required if the controller design does not make use of it.
        epsilon : float, optional
            The epsilon which might have been used to weight K.
        '''
        self.G1 = G1
        self.G2 = G2
        self.K = K
        self.epsilon = epsilon
        if Dc is None:
            Dc = np.zeros((K.shape[0], G2.shape[1]))
        self.Dc = Dc
        
        if not self.is_consistent():
            raise Exception('Controller parameters have incompatible dimensions.')


    def is_consistent(self):
        '''
        Verify that the dimensions of the matrices of the controller are consistent.

        Returns
        -------
        consistent : bool
            True if the dimensions are consistent, False otherwise.
        '''
        return self.G1.shape[0] == self.G1.shape[1] and  \
            self.G1.shape[0] == self.G2.shape[0] and \
            self.G1.shape[1] == self.K.shape[1] and \
            self.Dc.shape[0] == self.K.shape[0] and \
            self.Dc.shape[1] == self.G2.shape[1]


def construct_internal_model(freqs, dim_Y):
    '''
    Constructs a real internal model for a robust controller.

    Parameters
    ----------
    freqs : (, N) array_like
        The (real) frequencies of the reference and disturbance signals.
        Ordered so that 0=w_0<w_1<...<w_q (the list may or may not include the zero frequency).
    dim_Y : int
        The dimension of the output space Y of the system.

    Returns
    -------
    G1 : (M, M) array_like
        The matrix G1 of the controller.
    G2 : (M, dim_Y) array_like
        The matrix G2 of the controller.
    '''

    q = np.size(freqs)
    dim_Z = dim_Y * 2 * q
    if freqs[0] == 0:
        dim_Z -= dim_Y
    offset = 0
    G1 = np.zeros((dim_Z, dim_Z))
    # Alternatively, can replace -I_Y with an invertible square matrix.
    # The second block needs to remain zero.
    G2 = np.tile(np.concatenate((np.eye(dim_Y), np.zeros((dim_Y, dim_Y)))), (q, 1))

    if freqs[0] == 0:
        offset = 1
        # Alternatively, can replace -I_Y with an invertible square matrix.
        # The second block needs to remain zero.
        G2 = np.tile(np.concatenate((np.eye(dim_Y), np.zeros((dim_Y, dim_Y)))), (q - 1, 1))
        G2 = np.concatenate((np.eye(dim_Y), G2))

    for i in range(0, q - offset):
        idx = slice((2 * i + offset) * dim_Y, (2 * (i + 1) + offset) * dim_Y)
        G1[idx, idx] = freqs[i + offset] * np.bmat([[np.zeros((dim_Y, dim_Y)), np.eye(dim_Y)], [-np.eye(dim_Y), np.zeros((dim_Y, dim_Y))]])
    return G1, G2


def IMdim(freqs, dim_Y):
    '''
    # Compute the dimension of the internal model.

    Parameters
    ----------
    freqs : (, N) array_like
        The (real) frequencies of the reference and disturbance signals.
        Ordered so that 0=w_0<w_1<...<w_q (the list may or may not include the zero frequency).
    dim_Y : int
        The dimension of the output space Y of the system.

    Returns
    -------   
    dim_Z0 : int
        The dimension of the internal model.
    '''

    if freqs[0] == 0:
        return (2*len(freqs) - 1)*dim_Y
    else:
        return 2*len(freqs)*dim_Y

def IMstabilization_dissipative(freqs, Pvals):
    '''
    Stabilization of the internal model using dissipative design for the Low-Gain Robust Controller (see Paunonen IEEE TAC 2016, Section IV for details).

    Parameters
    ----------
    freqs : (, N) array_like
        The (real) frequencies of the reference and disturbance signals.
    Pvals : (N, M, P) array_like
        Values of the transfer function at the given frequencies.

    Returns
    -------
    K : (M1, N1) array_like
        The matrix K of the controller.
    '''
    choose_K0k = lambda Pval: np.linalg.pinv(Pval)

    q = np.size(freqs)
    dim_Y, dim_U = Pvals[0].shape
    dim_Z = 2 * q * dim_Y

    if freqs[0] == 0:
        dim_Z -= dim_Y
    offset = 0
    K = np.zeros((dim_U, dim_Z))

    if freqs[0] == 0:
        offset = 1
        K[:, 0:dim_Y] = np.real(choose_K0k(Pvals[0]))

    for i in range(0, q - offset):
        idx = slice((2 * i + offset) * dim_Y, (2 * (i + 1) + offset) * dim_Y)
        P_pi = choose_K0k(Pvals[i + offset])
        K[:, idx] = np.hstack((np.real(P_pi), np.imag(P_pi)))

    return K


def IMstabilization_general(freqs, Pvals, cG1, B1, IMstabmargin, IMstabmethod):
    '''
    Stabilization of the internal model pair (cG1,B1) using either LQR or pole placement.

    Parameters
    ----------
    freqs : (, N) array_like
        The (real) frequencies (w_k)_{k=0}^q of the reference and disturbance signals.
    Pvals : (N, M, P) array_like
        Values of the transfer function at the complex frequencies (i*w_k)_{k=0}^q.
    cG1 : (N1, M1) array_like
        The internal model for the frequencies 'freqs' and dim_Y
    B1 : (N2, M2) array_like
        The input matrix B1 of the internal model.
    IMstabmargin : float
        The desired stability margin for the internal model.
    IMstabmethod : string
        Stabilization of the internal model using either 'LQR' or 'poleplacement'.

    Returns
    -------
    K : (M1, N1) array_like
        The the stabilizing feedback K for the internal model so that cG1+B1*K is Hurwitz.

    Raises
    ------
    IMstabmethodexception : Exception 
       Thrown in case the internal model stabilization method is not 'LQR' or 'poleplacement'.
    '''
    dim_Y, dim_U = Pvals[0].shape

    if IMstabmethod == 'LQR':
        q = np.size(freqs)
        dim_Z = 2 * q * dim_Y - (freqs[0] == 0) * dim_Y
        return -lqr(cG1 + IMstabmargin * np.eye(cG1.shape[0], cG1.shape[1]), B1, 100 * np.eye(dim_Z), 0.001 * np.eye(dim_U))[0]
    elif IMstabmethod == 'poleplacement':
        # Necessary to get the same behaviour as in Matlab for
        # np.linspace(-1.1*IMstabmargin, -1*IMstabmargin, dim_Y)
        # in the case dim_Y = 1
        temp = (-np.linspace(IMstabmargin, 1.1 * IMstabmargin, dim_Y))[::-1]
        if freqs[0] == 0:
            t1 = np.dot(np.ones((2 * np.size(freqs) - 1, 1)), np.atleast_2d(temp))
            t2 = 1j * np.dot(np.atleast_2d(np.concatenate((freqs[::-1], -freqs[1:]))).T, np.ones((1, dim_Y)))
            target_eigs = (t1 + t2).flatten()
            return -place_poles(cG1, B1, target_eigs).gain_matrix
        else:
            t1 = np.dot(np.ones((2 * np.size(freqs), 1)), np.atleast_2d(temp))
            t2 = 1j * np.dot(np.atleast_2d(np.concatenate((freqs[::-1], -freqs))).T, np.ones((1, dim_Y)))
            target_eigs = (t1 + t2).flatten()
            return -place_poles(cG1, B1, target_eigs).gain_matrix
    else:
        raise Exception('Invalid IMstabmethod, choose either \'LQR\' or \'poleplacement\'')

def optimize_epsgain(A0, A1, A2, epsgain):
    '''
    Finds the positive real value :math:'\\varepsilon' for which
    :math:`A_0 + \\varepsilon  A_1 + \\varepsilon^2 A_2` has (roughly) the largest 
    stability margin. It is assumed that the stability margin has a unique 
    maximum among the provided 'epsgain', and the optimization algorithm 
    uses naive exhaustive search by starting from the smallest value of
    :math:`\\varepsilon` and stops once the stability margin ceases to increase.

    Parameters
    ----------
    A  : (M, M) array_like
        The first matrix.

    A1 : (M, M) array_like
        The second matrix.

    A2 : (M, M) array_like
        The third matrix.

    epsgain : (, L) array_like
        A list of possible values of :math:`\\varepsilon` or the limits of
        the optimization interval. If the list has a single element, this 
        value is returned. 
        If the list has two elements, the optimal :math:`\\varepsilon` is
        sought from on the interval :math:`[a,b]` determined by the values 
        in 'epsgain'. 
        If the list has more than three elements, the search is completed 
        among the values in 'epsgain'.

    Returns
    -------
    eps : float
        The (roughly) optimal value of :math:`\\varepsilon`.
    '''
    ee_cand = np.atleast_1d(epsgain)

    # A single element in 'epsgain', return this value.
    if np.size(ee_cand) == 1:
        return ee_cand[0]

    # Two elements in 'epsgain', search for the optimal value between
    # these values.
    if np.size(ee_cand) == 2:
        ee_cand = np.linspace(epsgain[0], epsgain[1], 20)

    # Three or more elements in 'epsgain', search for the optimal value 
    # among these values.

    old_stab_marg = 0
    marg_tol = 5e-4
    epsgain = ee_cand[0]

    # Find the epsilon with the largest stability margin.
    for ee in ee_cand:
        stab_margin = stability_margin(A0 + ee * A1 + np.square(ee) * A2)
        if stab_margin < old_stab_marg + marg_tol:
            break
        old_stab_marg = stab_margin
        epsgain = ee

    return epsgain


class LowGainRC(Controller):
    '''
    Construct a Low-Gain Robust Controller for a stable linear system.
    '''

    def __init__(self, sys, freqsReal, epsgain, Pvals, Dc=None):
        '''
        Parameters
        ----------
        sys : LinearSystem
            The controlled system.
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and disturbance signals.
        epsgain : array_like
            Can either be a scalar or a 1D array. If a scalar is given
            then this value will be used in the construction of the controller.
            If a 1D array is given, optimize_epsgain is used to (roughly)
            optimize the stability margin of the closed-loop system.
        Dc : (m,p) array_like, optional
            Feedthrough operator of the controller, required to be negative
            semidefinite. Assigned to be zero by default.
        Pvals : (N, M, P) array_like
             Values of the transfer function P(.) of the system at the complex frequencies (i*w_k)_{k=0}^q.

        Raises
        ------
        stabexception : Exception 
            Thrown if the original controlled system is not stable.
        '''
        if Dc is None:
            Dc = np.zeros((sys.B.shape[1],sys.C.shape[0]))
        stab_sys = sys.output_feedback(Dc)
        if not is_stable(stab_sys.A):
            raise Exception('System is not stable, cannot construct the controller')

        G1, G2tmp = construct_internal_model(freqsReal, Pvals[0].shape[0])
        G2 = -G2tmp
        K = IMstabilization_dissipative(freqsReal, Pvals)
        if scipy.sparse.issparse(stab_sys.A):
            A0 = np.bmat([[stab_sys.A.todense(), np.zeros((stab_sys.B.shape[0], G1.shape[1]))], [np.dot(G2, stab_sys.C), G1]])
        else:
            A0 = np.bmat([[stab_sys.A, np.zeros((stab_sys.B.shape[0], G1.shape[1]))], [np.dot(G2, stab_sys.C), G1]])
        A1 = np.bmat([[np.zeros((stab_sys.B.shape[0], A0.shape[1] - K.shape[1])), np.dot(stab_sys.B, K)],
                     [np.zeros((G2.shape[0], A0.shape[1] - K.shape[1])), np.dot(G2, np.dot(stab_sys.D, K))]])
        A2 = np.zeros((A0.shape[0], A0.shape[1]))

        eps = optimize_epsgain(A0, A1, A2, epsgain)
        print('Value of the low-gain parameter: %.3f' % eps)
        Controller.__init__(self, G1, G2, eps * K, None, eps)


class ObserverBasedRC_old(Controller):
    '''
    Construct an Observer-Based Robust Controller for a possibly unstable linear system.
    '''

    def __init__(self, sys, freqsReal, K21, L, IMstabmargin=0.5, IMstabmethod='LQR', CKRKvals=None):
        '''
        Parameters
        ----------
        sys : LinearSystem
            The controlled system.
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and
            disturbance signals.
        K21 : (N1, M1) array_like
             The matrix K2 which should be chosen so that A + B * K21
             is stable. 
        L : (N2, M2) array_like
             The matrix L1 which should be chosen so that A + L * C is stable. 
        IMstabmargin : float, optional
            The desired stability margin for the internal model. The default value is 0.5.
        IMstabmethod : string, optional
            Stabilization of the internal model using either 'LQR' or
            'poleplacement'. The default method is 'LQR'.
        CKRKvals : (q+1, n, m) array_like, optional
             Values of the function :math:`(C+D*K_{21})R(.,A+B*K_{21})` 
             evaluated at the complex frequencies :math:`(i*w_k)_{k=0}^q`.
             By default, the values are computed directly using the
             parameters :math:`(A,B,C,D)` of the system.
        '''

        # Values of the transfer function P_K(.) at the complex
        # frequencies (i*w_k)_{k=0}^q.
        PKvals = np.array(list(map(lambda freq1: sys.P_K(freq1, K21), 1j * freqsReal)))

        # The c in front of cG1, cG2 and cK signifies the internal model 
        # part of the controller
        cG1, cG2 = construct_internal_model(freqsReal, PKvals[0].shape[0])

        dim_X = sys.A.shape[0]
        dim_Y = PKvals[0].shape[0]
        dim_U = PKvals[0].shape[1]
        dim_Z0 = cG1.shape[0]
        q = np.size(freqsReal) # Number of frequencies
        if CKRKvals is None:
            CKRKvals = np.array(list(map(lambda freq: sys.CKRK(freq, K21), 1j * freqsReal)))

        # Construct the matrix B_1 based on PKvals and cG2
        B1 = np.zeros((dim_Z0, dim_U))
        H = np.zeros((dim_Z0, dim_X))
        if freqsReal[0] == 0:
            offset = 1
            G2comp = cG2[slice(0,dim_Y),:]
            # General version for an invertible matrix G_2^0
            B1[slice(0,dim_Y), :] =  np.matmul(G2comp,PKvals[0])
            H[slice(0,dim_Y), :] =  np.matmul(G2comp,CKRKvals[0])
            # Simpler version for the special case G_2^0=-I
            # B1[slice(0,dim_Y), :] =  -PKvals[0] 
            # H[slice(0,dim_Y), :] =  -CKRKvals[0] 
        else:
            offset = 0

        for i in range(0, np.size(freqsReal) - offset):
            idx = slice((2 * i + offset) * dim_Y, 
                    (2 * (i + 1) + offset) * dim_Y)
            
            idx_half = slice((2 * i + offset) * dim_Y, 
                    (2 * i + 1 + offset) * dim_Y)
            G2comp = cG2[idx_half,:]

            # General version for G_2^k = (-G_2^{k0},0)^T with G_2^{k0} invertible
            B1[idx, :] =  np.bmat([[np.matmul(G2comp,np.real(PKvals[i+offset]))], [np.matmul(G2comp,-np.imag(PKvals[i+offset]))]])
            H[idx, :] =  np.bmat([[np.matmul(G2comp,np.real(CKRKvals[i+offset]))], [np.matmul(G2comp,-np.imag(CKRKvals[i+offset]))]])
            # Simpler version for the special case G_2^i=(-I,0)^T
            # B1[idx, :] =  -np.bmat([[np.real(PKvals[i+offset])],
            #     [-np.imag(PKvals[i+offset])]]) 
            # H[idx, :] =  -np.bmat([[np.real(CKRKvals[i+offset])],
            #     [-np.imag(CKRKvals[i+offset])]]) 

        # print('Debug: Error in B1')
        # B1old = np.dot(H, sys.B) + np.dot(cG2, sys.D)
        # print(B1-B1old)
        # print('Debug: Error in H')
        # Hold = sp.linalg.solve_sylvester(cG1, -sys.A - np.dot(sys.B, K21), np.dot(cG2, sys.C + np.dot(sys.D, K21)))
        # print(np.amax(np.abs(Hold-H)))

        cK = IMstabilization_general(freqsReal, PKvals, cG1, B1, IMstabmargin, IMstabmethod)
        K2 = K21 + np.dot(cK, H)
        G11 = np.dot(sys.B + np.dot(L, sys.D), cK)
        G12 = sys.A + np.dot(sys.B, K2) + np.dot(L, sys.C + np.dot(sys.D, K2))
        dim1 = cG1.shape[0]
        dim2 = G11.shape[1] + G12.shape[1] - cG1.shape[1]
        G1 = np.bmat([[cG1, np.zeros((dim1, dim2))], [G11, G12]])
        G2 = np.vstack((cG2, -L))
        K = np.hstack((cK, K2))
        Controller.__init__(self, G1, G2, K, None, 1.0)


class ObserverBasedRC(Controller):
    '''
    Construct an Observer-Based Robust Controller for a possibly unstable linear system.
    '''

    def __init__(self, sys, freqsReal, K21, L, IMstabmargin=0.5, IMstabmethod='LQR'):
        '''
        Parameters
        ----------
        sys : LinearSystem
            The controlled system.
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and
            disturbance signals.
        K21 : (N1, M1) array_like
             The matrix K2 which should be chosen so that A + B * K21
             is stable. 
        L : (N2, M2) array_like
             The matrix L1 which should be chosen so that A + L * C is stable. 
        IMstabmargin : float, optional
            The desired stability margin for the internal model. The default value is 0.5.
        IMstabmethod : string, optional
            Stabilization of the internal model using either 'LQR' or
            'poleplacement'. The default method is 'LQR'.
        '''

        dim_X = sys.A.shape[0]
        dim_Y = sys.C.shape[0]
        dim_U = sys.B.shape[1]

        if dim_Y != dim_U:
            raise Exception("The system has an unequal number of inputs and outputs, the observer-based controller design cannot be completed (in this form).")

        # The c in front of cG1 and cG2 signifies the internal model 
        # part of the controller
        cG1, cG2 = construct_internal_model(freqsReal, dim_Y)

        dim_Z = cG1.shape[0]

        # Find H as the solution of G1*H=H*(A+B*K21)+G2*(C+D*K21) and define B1
        H = sp.linalg.solve_sylvester(cG1, -sys.A - np.dot(sys.B, K21), np.dot(cG2, sys.C + np.dot(sys.D, K21)))
        B1 = np.dot(H, sys.B) + np.dot(cG2, sys.D)

        K1 = IMstabilization_general(freqsReal, np.atleast_2d(dim_Y, dim_U), cG1, B1, IMstabmargin, IMstabmethod)

        K2 = K21 + np.dot(K1, H)
        G11 = np.dot(sys.B + np.dot(L, sys.D), K1)
        G12 = sys.A + np.dot(sys.B, K2) + np.dot(L, sys.C + np.dot(sys.D, K2))
        G1 = np.bmat([[cG1, np.zeros((dim_Z, dim_X))], [G11, G12]])
        G2 = np.vstack((cG2, -L))
        K = np.hstack((K1, K2))
        Controller.__init__(self, G1, G2, K, None, 1.0)


class DualObserverBasedRC_old(Controller):
    '''
    Construct a Dual Observer-Based Robust Controller for a possibly unstable linear system.
    '''

    def __init__(self, sys, freqsReal, K2, L1, IMstabmargin=0.5, IMstabmethod='LQR', RLBLvals=None):
        '''
        Parameters
        ----------
        sys : LinearSystem
            The controlled system.
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and
            disturbance signals.
        K2 : (N1, M1) array_like
             The matrix K2 which should be chosen so that A + B * K2 is stable. 
        L1 : (N2, M2) array_like
             The matrix L1 which should be chosen so that A + L1 * C is stable. 
        IMstabmargin : float, optional
            The desired stability margin for the internal model. The default
            value is 0.5.
        IMstabmethod : string, optional
            Stabilization of the internal model using either 'LQR' or
            'poleplacement'. The default method is 'LQR'.
        RLBLvals : (q+1, n, m) array_like, optional
             Values of the function :math:`R(.,A+L_1*C)(B+L_1*D)` evaluated 
             at the complex frequencies :math:`(i*w_k)_{k=0}^q`. By default,
             the values are computed directly using the parameters 
             :math:`(A,B,C,D)` of the system.

        '''

        # Values of the transfer function P_L(.) at the complex
        # frequencies  (i*w_k)_{k=0}^q.
        PLvals = np.array(list(map(lambda freq: sys.P_L(freq, L1), 1j * freqsReal)))

        # The c in front of cG1, cG2 and cK is just to take care of scoping.
        cG1, cG2 = construct_internal_model(freqsReal, PLvals[0].shape[0])
        cK = np.transpose(cG2)


        dim_X = sys.A.shape[0]
        dim_Y = PLvals[0].shape[0]
        dim_U = PLvals[0].shape[1]
        dim_Z0 = cG1.shape[0]
        q = np.size(freqsReal) # Number of frequencies

        if RLBLvals is None:
            RLBLvals = np.array(list(map(lambda freq: sys.RLBL(freq, L1), 1j * freqsReal)))
            
        # Construct the matrix C_1 based on PLvals and cG2
        C1 = np.zeros((dim_Y,dim_Z0))
        H = np.zeros((dim_X,dim_Z0))
        if freqsReal[0] == 0:
            offset = 1
            K1comp = cK[:,slice(0,dim_Y)]
            # General version for an invertible matrix K_1^0
            C1[:,slice(0,dim_Y)] =  np.matmul(PLvals[0],K1comp)
            H[:,slice(0,dim_Y)] =  np.matmul(RLBLvals[0],K1comp)
            # Simpler version for the special case K_1^0=-I
            # C1[:,slice(0,dim_Y)] =  -PLvals[0] 
            # H[:,slice(0,dim_Y)] =  -RLBLvals[0] 
        else:
            offset = 0

        for i in range(0, np.size(freqsReal) - offset):
            idx = slice((2 * i + offset) * dim_Y, 
                    (2 * (i + 1) + offset) * dim_Y)
            
            idx_half = slice((2 * i + offset) * dim_Y, 
                    (2 * i + 1 + offset) * dim_Y)
            K1comp = cK[:,idx_half]

            # General version for K_1^k = (-K_1^{k0},0) with K_1^{k0} invertible
            C1[:,idx] =  np.bmat([np.matmul(np.real(PLvals[i+offset]),K1comp), np.matmul(np.imag(PLvals[i+offset]),K1comp)])
            H[:,idx] =  np.bmat([np.matmul(np.real(RLBLvals[i+offset]),K1comp), np.matmul(np.imag(RLBLvals[i+offset]),K1comp)])
            # Simpler version for the special case K_1^i=(-I,0)
            # C1[:,idx] =  -np.bmat([np.real(PLvals[i+offset]),
            #     np.imag(PLvals[i+offset])]) 
            # H[:,idx] =  -np.bmat([np.real(RLBLvals[i+offset]), 
            # np.imag(RLBLvals[i+offset])]) 

        # print('Debug: Error in C1')
        # C1old = np.dot(sys.C, H) + np.dot(sys.D, cK)
        # print(C1-C1old)
        # print('Debug: Error in H')
        # Hold = sp.linalg.solve_sylvester(-sys.A - np.dot(L1, sys.C), cG1, np.dot(sys.B + np.dot(L1, sys.D), cK))
        # print(np.amax(np.abs(Hold-H)))

        cG2 = IMstabilization_general(freqsReal, PLvals, cG1.conj().T, C1.conj().T, IMstabmargin, IMstabmethod).conj().T
        L = L1 + np.dot(H, cG2)
        G11 = np.dot(cG2, sys.C + np.dot(sys.D, K2))
        G12 = sys.A + np.dot(sys.B, K2) + np.dot(L, sys.C + np.dot(sys.D, K2))
        dim1 = G12.shape[0]
        dim2 = cG1.shape[1] + G11.shape[1] - G12.shape[1]
        G1 = np.bmat([[cG1, G11], [np.zeros((dim1, dim2)), G12]])
        G2 = np.vstack((cG2, L))
        K = np.hstack((cK, -K2))
        Controller.__init__(self, G1, G2, K, None, 1.0)


class DualObserverBasedRC(Controller):
    '''
    Construct a Dual Observer-Based Robust Controller for a possibly unstable linear system.
    '''

    def __init__(self, sys, freqsReal, K2, L1, IMstabmargin=0.5, IMstabmethod='LQR'):
        '''
        Parameters
        ----------
        sys : LinearSystem
            The controlled system.
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and
            disturbance signals.
        K2 : (N1, M1) array_like
             The matrix K2 which should be chosen so that A + B * K2 is stable. 
        L1 : (N2, M2) array_like
             The matrix L1 which should be chosen so that A + L1 * C is stable. 
        IMstabmargin : float, optional
            The desired stability margin for the internal model. The default
            value is 0.5.
        IMstabmethod : string, optional
            Stabilization of the internal model using either 'LQR' or
            'poleplacement'. The default method is 'LQR'.

        '''

        dim_X = sys.A.shape[0]
        dim_Y = sys.C.shape[0]
        dim_U = sys.B.shape[1]

        if dim_Y != dim_U:
            raise Exception("The system has an unequal number of inputs and outputs, the observer-based controller design cannot be completed (in this form).")

        # The c in front of cG1 and cG2 signifies the internal model 
        # part of the controller
        cG1, cG2 = construct_internal_model(freqsReal, dim_Y)
        cK = np.transpose(cG2)

        dim_Z = cG1.shape[0]

        # Find H as the solution of H*G1=(A+L1*C)*H+(B+L1*D)*K and define C1
        H = sp.linalg.solve_sylvester(-sys.A - np.dot(L1, sys.C), cG1, np.dot(sys.B + np.dot(L1, sys.D), cK))
        C1 = np.dot(sys.C, H) + np.dot(sys.D, cK)

        cG2 = IMstabilization_general(freqsReal, np.atleast_2d(dim_Y, dim_U), cG1.conj().T, C1.conj().T, IMstabmargin, IMstabmethod).conj().T
        L = L1 + np.dot(H, cG2)
        G11 = np.dot(cG2, sys.C + np.dot(sys.D, K2))
        G12 = sys.A + np.dot(sys.B, K2) + np.dot(L, sys.C + np.dot(sys.D, K2))
        G1 = np.bmat([[cG1, G11], [np.zeros((dim_X, dim_Z)), G12]])
        G2 = np.vstack((cG2, L))
        K = np.hstack((cK, -K2))
        Controller.__init__(self, G1, G2, K, None, 1.0)


class PassiveRC(Controller):
    '''
    Construct a Passive Robust Controller for a stable impedance passive linear system.
    '''

    def __init__(self, freqsReal, dim_Y, epsgain, sys, Dc=None, Pvals=None):
        '''
        Parameters
        ----------
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and disturbance signals.
        dim_Y : integer
            Dimension of the output space.
        epsgain : array_like
            Can either be a scalar or a 1D array. If a scalar is given
            then this value will be used in the construction of the controller.
            If a 1D array is given, optimize_epsgain is used to (roughly)
            optimize the stability margin of the closed-loop system.
        sys : LinearSystem
            The system to construct the controller for. Only used in
            optimizing the gain parameter 'eps'.
        Dc : (m,p) array_like, optional
            Feedthrough operator of the controller, required to be negative
            semidefinite. Assigned to be zero by default.
        Pvals : (q+1, dim_Y, dim_U) array_like, optional
             Values of the transfer function P at the complex frequencies 
             (i*w_k)_{k=0}^q. [Not used in the construction at the moment]

        Raises
        ------
        stabexception : Exception 
            Thrown if the original controlled system is not stable.

        Note that the construction routine does not test the passivity 
        of the system.
        '''
        if Dc is None:
            Dc = np.zeros((sys.B.shape[1],sys.C.shape[0]))
        stab_sys = sys.output_feedback(Dc)
        if not is_stable(stab_sys.A):
            raise Exception('System is not stable, cannot construct the controller')

        G1, G2tmp = construct_internal_model(freqsReal, dim_Y)
        G2 = -G2tmp
        K = G2tmp.conj().T
        if scipy.sparse.issparse(sys.A):
            A0 = np.bmat([[stab_sys.A.todense(), np.zeros((stab_sys.B.shape[0], G1.shape[1]))], [np.dot(G2, stab_sys.C), G1]])
        else:
            A0 = np.bmat([[stab_sys.A, np.zeros((stab_sys.B.shape[0], G1.shape[1]))], [np.zeros((G2.shape[0], stab_sys.A.shape[1])), G1]])
        A1 = np.bmat([[np.zeros(stab_sys.A.shape), np.dot(stab_sys.B, K)],
                     [np.dot(G2, stab_sys.C), np.zeros(G1.shape)]])
        A2 = np.bmat([[np.zeros(stab_sys.A.shape), np.zeros((stab_sys.A.shape[0],G1.shape[1]))],
                     [np.zeros((G2.shape[0], stab_sys.A.shape[1])), np.dot(G2, np.dot(stab_sys.D, K))]])
        eps = optimize_epsgain(A0, A1, A2, epsgain)
        print('Value of the low-gain parameter: %.3f' % eps)
        Controller.__init__(self, G1, eps * G2, eps * K, Dc, eps)


class ApproximateRC(Controller):
    '''
    Construct an Approximate Robust Controller for a stable linear system (for details, see Humaloja, Kurula and Paunonen IEEE TAC 2019. [Work in progress].
    '''

    def __init__(self, freqs, N, Ny, P2vals, PN, epsgain, sys=None):
        '''
        Parameters
        ----------
        freqs : (, N) array_like
            The frequencies of the reference and disturbance signals. 
        N : int
            Size of the 'full' approximation of the output space Y
        Ny : int
            The size of the intended approximated output space Y_N, needs to be
            1 <= Ny <= N
        P2vals : (N, M, P) array_like
             Values of the transfer function P2(.) at the given frequencies.
        PN : (N1, M1) array_like
            The projection matrix from the 'full' output space Y to the approximate output space Y_N
        epsgain : array_like
            Can either be a scalar or a 1D array. If a scalar is given
            then this value will be used in the construction of the controller.
            If a 1D array is given, optimize_epsgain is used to (roughly)
            optimize the stability margin of the closed-loop system.
        '''
        dim_W = np.size(freqs)
        # TODO: Incorrect sizes of the output Y and Y_N spaces.
        dim_U = N
        dim_Y = Ny
        G1 = np.diag(np.dot(np.ones((dim_Y, 1)), np.atleast_2d(freqs)).flatten(order='F'))
        G2 = np.tile(np.dot(-np.eye(dim_Y), PN), (dim_W, 1))
        K = np.zeros((dim_U, dim_Y * dim_W), 'complex')
        for i in range(0, dim_W):
            idxran = slice(i * dim_Y, (i + 1) * dim_Y)
            A = PN.conj().T
            B = np.dot(PN, PN.conj().T)
            K[:, idxran] = np.linalg.solve(P2vals[i], np.dot(A, np.linalg.solve(np.dot(B, B.conj().T), B)))

        # Optimize the value of the low-gain parameter eps
        if sys is not None:
            A = np.bmat([[sys.A, np.zeros((sys.A.shape[0], G1.shape[1]))], [np.dot(G2, sys.C), G1]])
            B = np.bmat([[np.zeros((sys.B.shape[0], A.shape[1] - K.shape[1])), np.dot(sys.B, K)],
                         [np.zeros((G2.shape[0], A.shape[1] - K.shape[1])), np.dot(G2, np.dot(sys.D, K))]])
            eps = optimize_epsgain(A, B, epsgain)
        else:
            eps = np.atleast_1d(epsgain)[0]
        Controller.__init__(self, G1, G2, eps * K, None, eps)


class ObserverBasedROMRC(Controller):
    '''
    Construct a Reduced Order Robust Controller for parabolic systems.
    '''

    def __init__(self, sysApprox, freqsReal, alpha1, alpha2, R1, R2, Q0, Q1, Q2, ROMorder):
        '''
        Parameters
        ----------
        sysApprox : LinearSystem
            The Galerkin approximation (A^N,B^N,C^N,D) of the control system for the controller design.
        freqsReal : (, N) array_like
            The (real) frequencies (w_k)_{k=0}^q of the reference and disturbance signals.
        alpha1, alpha2 : integer
            Design parameters to determine the closed-loop stability margin. Required to be positive.
        R1, R2, Q0, Q1, Q2 : integer/(N1, M1) array_like
            TODO
        ROMorder
            order of the reduced-order observer in the controller.
            The model reduction step can be skipped by setting 'ROMorder=None'
        '''

        AN = sysApprox.A
        BN = sysApprox.B
        CN = sysApprox.C
        D = sysApprox.D

        dim_X = AN.shape[0]
        dim_Y = CN.shape[0]
        dim_U = BN.shape[1]

        # Check the consistency of the controller design parameters

        alpha1_check = np.isreal(alpha1) and np.ndim(alpha1) == 0 and alpha1 > 0
        alpha2_check = np.isreal(alpha2) and np.ndim(alpha2) == 0 and alpha2 > 0

        R1_check = np.allclose(R1, np.conj(R1).T) and np.all(np.linalg.eigvals(R1) > 0)
        R2_check = np.allclose(R2, np.conj(R2).T) and np.all(np.linalg.eigvals(R2) > 0)

        if not alpha1_check or not alpha2_check:
            raise Exception('"alpha1" and "alpha2" need to be positive!')
        elif not R1_check or not R2_check:
            raise Exception('"R1" and "R2" need to be positive definite')

        # Construct the internal model
        cG1, cG2 = construct_internal_model(freqsReal, dim_Y)
        dim_Z0 = cG1.shape[0]


        # If R1, R2, Q0, Q1, or Q2 have scalar values, these are interpreted as
        # "scalar times identity".
        if dim_Y > 1 and np.ndim(R1) == 0:
            R1 = R1*np.eye(dim_Y)
        
        if dim_Y > 1 and np.ndim(R2) == 0:
            R2 = R2*np.eye(dim_Y)

        if dim_Z0 > 1 and np.ndim(Q0) == 0:
            Q0 = Q0*np.eye(dim_Z0)

        if dim_X > 1 and np.ndim(Q1) == 0:
            Q1 = Q1*np.eye(dim_X)

        if dim_X > 1 and np.ndim(Q2) == 0:
            Q2 = Q2*np.eye(dim_X)



        # Check the consistency of the dimensions of R1, R2, Q0, Q1, and Q2
        if not R1.shape == (dim_Y, dim_Y):
            raise Exception('Dimensions of "R1" are incorrect! (should be [dimY,dimY]).')

        if not R2.shape == (dim_U, dim_U):
            raise Exception('Dimensions of "R1" are incorrect! (should be [dimU,dimU]).')

        if not Q0.shape[1] == dim_Z0:
            raise Exception('Dimensions of "Q0" are incorrect!')

        if not Q1.shape[0] == dim_X:
            raise Exception('Dimensions of "Q1" are incorrect!')

        if not Q2.shape[1] == dim_X:
            raise Exception('Dimensions of "Q2" are incorrect!')


        # Form the extended system (As,Bs)
        As = np.bmat([[cG1, np.dot(cG2, CN)], [np.zeros((dim_X, dim_Z0)), AN]])
        Bs = np.bmat([[np.dot(cG2, D)], [BN]])

        Qs = sp.linalg.block_diag(Q0, Q2)


        # Stabilize the pairs (CN,AN+alpha1) and (As+alpha2,Bs) using LQR/LQG design

        # Stabilize the pair (CN,AN+alpha1)
        t1 = time.time()
        B_ext = np.bmat([[np.conj(CN).T, np.zeros((dim_X, dim_X))]])
        # print("B_ext is")
        # print(B_ext.shape)
        # print(B_ext)
        S_ext = np.bmat([[np.zeros((dim_X, dim_Y)), Q1]])
        # print("S_ext is")
        # print(S_ext.shape)
        # print(S_ext)
        R1_ext = scipy.linalg.block_diag(R1, -np.eye(dim_X))
        # print("R1_ext is")
        # print(R1_ext.shape)
        # print(R1_ext)
        
        # Following the matlab version use of icare we have E = I and G = 0.
        # q = np.zeros(((AN + alpha1*np.eye(dim_X)).shape[0], (AN + alpha1*np.eye(dim_X)).shape[1]))
        # X = scipy.linalg.solve_continuous_are((AN + alpha1*np.eye(dim_X)), B_ext, q, R1_ext, s=S_ext)
        # E = I
        # E = np.eye(X.shape[1])
        # K = R^-1(B'XE + S')
        # L_ext_adj  = np.dot(np.linalg.inv(R1_ext), np.dot(np.dot(B_ext.T, X), E) + S_ext.T)
        # print(L_ext_adj)
        # L = EIG(A+G*X*E-B*K,E)
        # G = 0 -> G*X*E = zeros(dim(X,1), dim(E,2))
        # evals = np.linalg.eigvals((AN + alpha1*np.eye(dim_X)) + np.zeros((X.shape[0], E.shape[1])) - np.dot(B_ext, L_ext_adj))
        # [X, K, L] = [~,L_ext_adj,evals] = icare((AN+alpha1*eye(dimX))',B_ext,0,R1_ext,S_ext,[],[]);

        # Using the control library:
        q = np.zeros(((AN + alpha1*np.eye(dim_X)).shape[0], (AN + alpha1*np.eye(dim_X)).shape[1]))
        e = np.eye(q.shape[0], q.shape[1])
        # print("A is")
        # print((AN + alpha1*np.eye(dim_X)).conj().T.shape)
        # print((AN + alpha1*np.eye(dim_X)).conj().T)
        X1, evals, L_ext_adj = control.care(A=(AN + alpha1*np.eye(dim_X)).conj().T, B=B_ext, Q=q, R=R1_ext, S=S_ext, E=e)
        # print("L_ext_adj is")
        # print(L_ext_adj.shape)
        # print(L_ext_adj)
        # L_ext_adj  = np.dot(np.linalg.inv(R1_ext), np.dot(np.dot(B_ext.T, X), e) + S_ext.T)
        #print(L_ext_adj)
        # X, evals, L_ext_adj, evals = control.matlab.care((AN+alpha1*eye(dimX)),B_ext,0,R1_ext,S_ext)
        # print("testing")
        
        # L = -L_ext_adj(1:dimY,:)';
        L = -np.conj(L_ext_adj[0:dim_Y, :]).T
        # print("This is L")
        # print(L.shape)
        # print(L)
        t2 = time.time()
        t = t2 - t1
        print('Stabilization of the pair (CN,AN+alpha1) took %f seconds' % t)
        
        if np.amax(np.real(evals)) >= 0:
            raise Exception('Stabilization of the pair (CN,AN+alpha1) failed!')
        
        # Stabilize the pair (As+alpha2,Bs)
        t3 = time.time()
        Bs_ext = np.bmat([[Bs, np.zeros((dim_Z0 + dim_X, dim_Z0 + dim_X))]])
        # print("Bs_ext is")
        # print(Bs_ext.shape)
        # print(Bs_ext)
        Ss_ext = np.bmat([[np.zeros((dim_Z0 + dim_X, dim_Y)), np.conj(Qs).T]])
        # print("Ss_ext is")
        # print(Ss_ext.shape)
        # print(Ss_ext)
        R2_ext = scipy.linalg.block_diag(R2, -np.eye(dim_Z0 + dim_X))
        # print("R2_ext is")
        # print(R2_ext.shape)
        # print(R2_ext)
        
        # Following the matlab version use of icare we have E = I and G = 0.
        # qs = np.zeros(((As + alpha2*np.eye(dim_Z0 + dim_X)).shape[0], (As + alpha2*np.eye(dim_Z0 + dim_X)).shape[1]))
        # Xs = scipy.linalg.solve_continuous_are((As + alpha2*np.eye(dim_Z0 + dim_X)), Bs_ext, qs, R2_ext, s=Ss_ext)
        # K = R^-1(B'XE + S')
        # Es = I
        # Es = np.eye(Xs.shape[1])
        # K_ext  = np.dot(np.linalg.inv(R2_ext), np.dot(np.dot(Bs_ext.T, Xs), Es) + Ss_ext.T)
        # L = EIG(A+G*X*E-B*K,E)
        # G = 0 -> G*X*E = zeros(dim(X,1), dim(E,2))
        # evals = np.linalg.eigvals((As + alpha2*np.eye(dim_Z0 + dim_X)) + np.zeros((Xs.shape[0], Es.shape[1])) - np.dot(Bs_ext, K_ext))
        # [X, K, L] = [~,K_ext,evals] = icare(As+alpha2*eye(dimZ0+dimX),Bs_ext,0,R2_ext,Ss_ext,[],[]);

        # Using the control library:
        qs = np.zeros(((As + alpha2*np.eye(dim_Z0 + dim_X)).shape[0], (As + alpha2*np.eye(dim_Z0 + dim_X)).shape[1]))
        es = np.eye(qs.shape[0], qs.shape[1])
        X2, evals, K_ext = control.care(A=(As + alpha2*np.eye(dim_Z0 + dim_X)), B=Bs_ext, Q=qs, R=R2_ext, S=Ss_ext, E=es)
        # Es = np.eye(Xs.shape[1])
        # K_ext  = np.dot(np.linalg.inv(R2_ext), np.dot(np.dot(Bs_ext.T, X), Es) + Ss_ext.T)
        
        # K = -K_ext(1:dimU,:);
        K = -K_ext[0:dim_U, :]
        # print("This is K")
        # print(K.shape)
        # print(K)
        t4 = time.time()
        t = t4 - t3
        print('Stabilization of the pair (As+alpha2,Bs) took %f seconds' % t)

        if np.amax(np.real(evals)) >= 0:
            raise Exception('Stabilization of the pair (As+alpha2,Bs) failed!')


        # Decompose the control gain K into K=[K1N,K2N]
        K1N = K[:, 0:dim_Z0]
        K2N = K[:, dim_Z0:]


        # Complete the model reduction step of the controller design. If the model
        # reduction fails (for example due to too high reduction order), the
        # controller design is by default completed without the model reduction.
        if ROMorder is not None:
            try:
                t5 = time.time()
                # rsys = balred(ss(AN+L*CN,[BN+L*D,L],K2N,zeros(dimU,dimU+dimY)),ROMorder);
                rsys = control.balred(control.ss(AN + np.dot(L, CN), np.bmat([[BN + np.dot(L, D), L]]), K2N, np.zeros((dim_U, dim_U + dim_Y))), ROMorder, method='matchdc')
                # print(rsys)
                t6 = time.time()
                t = t6 - t5
                print('Model reduction step took %f seconds' % t)
                
                ALr = rsys.A
                Br_full = rsys.B
                BLr = Br_full[:, 0:dim_U]
                Lr = Br_full[:, dim_U:]
                K2r = rsys.C
            except:
                print('Model reduction step failed! Modify "ROMorder", or check that "balred" is available. Proceeding without model reduction (with option "ROMorder=None")')

                ALr = AN + np.dot(L, CN)
                BLr = BN + np.dot(L, D)
                Lr = L
                K2r = K2N
        else:
            print('Constructing the controller without model reduction.')
            
            ALr = AN + np.dot(L, CN)
            BLr = BN + np.dot(L, D)
            Lr = L
            K2r = K2N


        # Construct the controller matrices (G1,G2,K).
        G1 = np.bmat([[cG1, np.zeros((dim_Z0, ALr.shape[0]))], [np.dot(BLr, K1N), ALr + np.dot(BLr, K2r)]])
        G2 = np.bmat([[cG2], [-Lr]])
        K = np.bmat([[K1N, K2r]])
        Dc = np.zeros((dim_U, dim_Y))
        Controller.__init__(self, G1, G2, K, Dc, 1.0)