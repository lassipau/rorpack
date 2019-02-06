'''
Contains functions for constructing sparse discrete laplacians.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
from scipy.sparse import spdiags, kron, eye
from rorpack.system import LinearSystem


def laplacian_1d(N, B):
    '''
    Sparse discrete 1D Laplacian with Dirichlet/Neumann boundary conditions.

    N is the x-direction, and B is a 1D array specifying the boundary
    conditions.
    TODO: not quite sure about handling of D and N here
    '''
    assert B.shape == (2,)
    assert N >= 1

    ex = np.ones((1, N))
    Dxx = spdiags(np.vstack((-ex, 2 * ex, -ex)), [-1, 0, 1], N, N).tolil()

    if B[0] == 'N':
        Dxx[0, 1] -= 1
    if B[1] == 'N':
        Dxx[N-1, N-2] -= 1

    return Dxx

def diffusion_op_1d(spgrid, cfun, BCtype):
    '''
    Sparse discrete 1D diffusion operator A defined by (Au)(x)=(c(x)*u'(x))' 
    with Dirichlet or Neumann boundary conditions.

    Parameters:
    ---------
    spgrid : array_like
        The discretized (uniform) spatial grid on the interval [0,L] so that
        spgrid[0]=0 and spgrid[end]=L, 
    cfun : function
        Defines the spatially dependent coefficient function c(x)
    BCtype : string
        Boundary condition configuration, either
        'DD' for Dirichlet at x=0, Dirichlet at x=1
        'DN' for Dirichlet at x=0, Neumann at x=1
        'ND' for Neumann at x=0, Dirichlet at x=1
        'NN' for Neumann at x=0, Neumann at x=1

    Returns
    ---------
    Dop : (N, M) array_like
        The square matrix representing the diffusion operator.
    spgrid : (1,N) array_like
        An adjusted spatial grid: 
    '''

    N = np.size(spgrid)
    assert N >= 1
    h = spgrid[1]-spgrid[0]

    # We use the approximation:  
    # (c(x)u'(x))'(x) ~ 1/h^2*(c(x+h/2)*(u(x+h)-u(x))-c(x-h/2)*(u(x)-u(x-h))

    # points x_k+h/2
    spmidpoints = 1/2.0*(spgrid[0:-1]+spgrid[1:])
    cmid = cfun(spmidpoints)


    # Diffusion operator base (Neumann-Neumann case)
    cminus = np.concatenate((cmid,np.atleast_1d(0)))
    cplus = np.concatenate((np.atleast_1d(0),cmid))
    ex = np.ones((1, N))
    Dop = spdiags(np.vstack((cminus, -(cminus+cplus), cplus)), [-1, 0, 1], N, N).tolil()
    Dop[0,0] = -(cfun(spgrid[0])+cfun(spgrid[1]))
    Dop[0,1] = cfun(spgrid[0])+cfun(spgrid[1])
    Dop[-1,-1] = -(cfun(spgrid[-1])+cfun(spgrid[-2]))
    Dop[-1,-2] = cfun(spgrid[-1])+cfun(spgrid[-2])
    Dop = 1/(np.square(h))*Dop

    # Neumann-Neuman BCs
    if BCtype is 'NN':
        # No need to change the grid
        spgrid = spgrid
    elif BCtype is 'ND':
        spgrid = spgrid[0:-1]
        Dop = Dop[0:-1,0:-1]
    elif BCtype is 'DN':
        spgrid = spgrid[1:]
        Dop = Dop[1:,1:]
    elif BCtype is 'DD':
        spgrid = spgrid[1:-1]
        Dop = Dop[1:-1,1:-1]
    else:
        raise Exception('Unrecognised boundary condition types.')

    return Dop, spgrid

def laplacian_2d(N, M, B):
    '''
    Sparse discrete 2D Laplacian on a rectangular grid with Dirichlet/Neumann
    boundary conditions.

    N and M are the x- and y-directions, and B is a 2x2 matrix for specifying
    the boundary conditions.
    '''
    assert B.shape == (2, 2)
    assert N >= 1
    assert M >= 1

    ex = np.ones((1, N))
    ey = np.ones((1, M))
    Dxx = spdiags(np.vstack((-ex, 2 * ex, -ex)), [-1, 0, 1], N, N).tolil()
    Dyy = spdiags(np.vstack((-ey, 2 * ey, -ey)), [-1, 0, 1], M, M).tolil()

    if B[0, 0] == 'N':
        Dxx[0, 0] = 1
    if B[0, 1] == 'N':
        Dxx[-1, -1] = 1
    if B[1, 0] == 'N':
        Dyy[0, 0] = 1
    if B[1, 1] == 'N':
        Dyy[-1, -1] = 1

    return kron(Dyy, eye(N)) + kron(eye(M), Dxx)
