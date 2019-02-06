'''Various utility routines used in the RORPack library.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numbers
import numpy as np
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt


def stability_margin(A):
    '''
    Returns the stability margin of a linear system with a system matrix :math:`A`.

    Parameters
    ----------
    A : (M, M) array_like
        The system matrix of the linear system.

    Returns
    -------
    stabmarg : float
        The stability margin of the system. Negative if the system is unstable.
    '''
    if scipy.sparse.issparse(A):
        return -np.amax(np.real(np.linalg.eig(A.todense())[0])) 
    else:
        return -np.amax(np.real(np.linalg.eig(A)[0]))


def is_stable(A):
    '''
    Checks the stability of a linear system with a system matrix :math:`A`.

    Parameters
    ----------
    A : (M, M) array_like
        The system matrix of the linear system.

    Returns
    -------
    stable : bool
        True if the system matrix :math:`A` is Hurwitz, False otherwise.
    '''
    return stability_margin(A) > 0.0


def stability_margin_and_stiffness_ratio(A):
    '''
    Computes the stability margin and the stiffness ratio of a linear system with a system matrix `A`.

    Parameters
    ----------
    A : (M, M) array_like
        The system matrix of the linear system.

    Returns
    -------
    stabmarg : float
        The system matrix of the linear system.

    stiffnessrat : float
        The stiffness ratio. If the system is unstable, this is None.
    '''
    re_eig = np.real(np.linalg.eig(A)[0])
    stability_margin = -np.amax(re_eig)
    if stability_margin < 0:
        return (stability_margin, None)
    return stability_margin, np.amax(np.abs(re_eig)) / np.amin(np.abs(re_eig))


def vectorize(f):
    '''
    Given a function `f` that produces either scalars or 1D arrays for 
    scalar inputs, the routine 'vectorize' returns a new function that 
    (1) produces 1D arrays for scalar inputs and
    (2) produces 2D matrices for 1D array inputs (in a way that the number 
    of rows in the latter is the size in the of the input array).

    Parameters
    ----------
    f : function
        The function to vectorize.

    Returns
    -------
    vf : function
        The vectorized function.
    '''
    def go(t):
        if isinstance(t, numbers.Real):
            return np.atleast_1d(f(t))
        temp = list(map(f, t))
        if np.size(temp[0]) == 1:
            return np.array(temp)
        else:
            return np.column_stack(temp)
    return lambda t: go(t)


def savemat(A, name):
    '''
    Saves the array `A` into a MATLAB-style .mat file with the given name.

    Parameters
    ----------
    A : (M, N) array_like
        The array to be saved.

    name : string
        The name of the file to which `A` is saved.
    '''
    scipy.io.savemat(name + '.mat', dict(A=A))


def lqr(A, B, Q, R):
    """
    Computes the stabilizing feedback :math:`K` for the pair :math:`(A,B)` 
    based on the infinite-time linear quadratic optimal control.

    Parameters
    ----------
    A : (M, M) array_like
        The system matrix.

    B : (M, N) array_like
        The input matrix.

    Q : (M, M) array_like
        The state cost matrix.

    R : (N, N) array_like
        The control cost matrix.

    Returns
    -------
    K : (N, M) array_like
        The state feedback matrix.

    S : array_like
        The solution of the Riccati equation.
   
    e : array_like
        Eigenvalues of the closed-loop matrix :math:`A+BK`.
    """
    # Solve the Riccati equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # Compute the LQR gain based on X.
    K = np.linalg.solve(R, np.dot(B.T, X))
    S = X
    e = scipy.linalg.eig(A - np.dot(B, K))[0]

    return K, S, e


def fix_freqs(freqs):
    """
    Produces a list of unique positive real frequencies from a list of possibly complex frequencies, negative frequencies, and frequencies with multiplicities.

    Parameters
    ----------
    freqs : (, M) array_like
        The list of input frequencies.

    Parameters
    ----------
    freqsReal : (, M) array_like
        The list of output frequencies.
    """
    # Test that the frequencies are either all real or all complex.
    all_real = all(np.isreal(freqs))
    all_complex = all(freq == 0j or freq != freq.conjugate() for freq in freqs)
    if not (all_real or all_complex):
        raise Exception('A combination of real and complex freqs is not allowed')
    return np.unique(np.abs(freqs))
