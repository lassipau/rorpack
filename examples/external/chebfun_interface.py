'''
Provides an interface for MATLAB/Chebfun computations in the PDE examples.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
import scipy.io
import scipy.linalg
import matlab.engine


def heat_1d_2_Pvals(cfun,freqsReal):
    '''
    Computes the values P(i*w_k) of the transfer function in the example 'heat_1d_2' in the case of the "Low-Gain Robust Controller"

    Parameters
    ----------
    cfun : string
        The thermal diffusivity function (given as string in Matlab syntax, a function of 'x')
    freqsReal : (, N) array_like
        The (real) frequencies (w_k)_{k=0}^q of the reference and
        disturbance signals.

    Returns
    ----------
    Pvals : (q+1,p,m) array
        The values P(i*w_k) for k=0..q
    '''
    print('Thermal diffusivity function: ' + cfun)

    print('Computing the transfer function values P(iw_k) with Chebfun/Matlab.')

    print('Starting Matlab engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath('external')
    q = freqsReal.size
    Pvals = np.zeros((q,1,1),dtype=np.complex)
    # Guarantee that the zero frequency is evaluated as a real number
    offset = 0
    if freqsReal[0] == 0:
        print('Computing frequency w_k=0')
        Pvals[0] = eng.MATLAB_heat_1d_2('Pvals',cfun,0.0,0,0,0)
        offset = 1
    for ind in range(offset,q):
        print('Computing frequency w_k=%f' % freqsReal[ind])
        Pvals[ind] = eng.MATLAB_heat_1d_2('Pvals',cfun,np.complex(1j*freqsReal[ind]),0,0,0)
    print('Computation of P(iw_k) complete!')
    return Pvals

def heat_1d_2_PKvals(cfun,freqsReal,K21fun,spgrid):
    '''
    Computes the values P_K(i*w_k) of the transfer function and
    (C+D*K21)R(i*w_k,A+B*K21) in the example 'heat_1d_2' in the case 
    of the "Observer-Based Robust Controller"

    Parameters
    ----------
    cfun : string
        The thermal diffusivity function (given as string in Matlab syntax, a   function of 'x')
    freqsReal : (, N) array_like
        The (real) frequencies (w_k)_{k=0}^q of the reference and
        disturbance signals.
    K21fun : string
        A function of 'x' describing the state feedback operator K21, given in Matlab syntax
    spgrid : (,N) array_like
        Spatial grid of the discretization of [0,1]

    Returns
    ----------
    PKvals : (q+1,p,m) array
        The values P_K(i*w_k) for k=0..q
    CKRKvals : (q+1,p,m) array
        The values (C+D*K21)R(i*w_k,A+B*K21) for k=0..q
    '''
    spgrid_ML = matlab.double(spgrid.tolist())

    print('Thermal diffusivity function: ' + cfun)
    print('K21 function: ' + K21fun)

    print('Computing the transfer function values P_K(iw_k) with Chebfun/Matlab.')

    print('Starting Matlab engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath('external')
    q = freqsReal.size
    PKvals = np.zeros((q,1,1),dtype=np.complex)
    N = spgrid.size
    CKRK = np.zeros((q,1,N),dtype=np.complex)
    # Guarantee that the zero frequency is evaluated as a real number
    offset = 0
    if freqsReal[0] == 0:
        print('Computing frequency w_k=0')
        tmp = eng.MATLAB_heat_1d_2('PKvals',cfun,0.0,spgrid_ML,K21fun,0,nargout=3)

        # Process the data returned by the Matlab engine
        PKvals[0] = tmp[0]
        CKRKtmp = tmp[1]
        CKRKpy = np.array(CKRKtmp._data.tolist())
        CKRK[0,] = CKRKpy.reshape((1,N))
        offset = 1
    for ind in range(offset,q):
        print('Computing frequency w_k=%f' % freqsReal[ind])
        tmp = eng.MATLAB_heat_1d_2('PKvals',cfun,np.complex(1j*freqsReal[ind]),spgrid_ML,K21fun,0,nargout=3)

        # Process the data returned by the Matlab engine
        PKvals[ind] = tmp[0]
        CKRKtmp = tmp[1]
        CKRKpyRE = np.real(CKRKtmp)
        CKRKpyIM = np.imag(CKRKtmp)
        CKRK[ind,] = CKRKpyRE.reshape((1,N)) + 1j*CKRKpyIM.reshape((1,N))

    print('Computation of P_K(iw_k) and CKRK[k] complete!')
    return PKvals, CKRK


def heat_1d_2_PLvals(cfun,freqsReal,L1fun,spgrid):
    '''
    Computes the values P_L(i*w_k) of the transfer function and
    R(i*w_k,A+L1*C)(B+L1*D) in the example 'heat_1d_2' in the case 
    of the "Dual Observer-Based Robust Controller"


    Parameters
    ----------
    cfun : string
        The thermal diffusivity function (given as string in Matlab syntax, a   function of 'x')
    freqsReal : (, N) array_like
        The (real) frequencies (w_k)_{k=0}^q of the reference and
        disturbance signals.
    L1fun : string
        A function of 'x' describing the output injection operator L1, given in Matlab syntax
    spgrid : (,N) array_like
        Spatial grid of the discretization of [0,1]

    Returns
    ----------
    PLvals : (q+1,p,m) array
        The values P_L(i*w_k) for k=0..q
    RLBLvals : (q+1,p,m) array
        The values R(i*w_k,A+L1*C)(B+L1*D) for k=0..q
    '''
    spgrid_ML = matlab.double(spgrid.tolist())

    print('Thermal diffusivity function: ' + cfun)
    print('L1 function: ' + L1fun)

    print('Computing the transfer function values P_L(iw_k) with Chebfun/Matlab.')

    print('Starting Matlab engine...')
    eng = matlab.engine.start_matlab()
    eng.addpath('external')
    q = freqsReal.size
    PLvals = np.zeros((q,1,1),dtype=np.complex)
    N = spgrid.size
    RLBL = np.zeros((q,N,1),dtype=np.complex)
    # Guarantee that the zero frequency is evaluated as a real number
    offset = 0
    if freqsReal[0] == 0:
        print('Computing frequency w_k=0')
        tmp = eng.MATLAB_heat_1d_2('PLvals',cfun,0.0,spgrid_ML,0,L1fun,nargout=3)

        # Process the data returned by the Matlab engine
        PLvals[0] = tmp[0]
        RLBLtmp = tmp[2]
        RLBLpy = np.array(RLBLtmp._data.tolist())
        RLBL[0,] = RLBLpy.reshape((N,1))
        offset = 1
    for ind in range(offset,q):
        print('Computing frequency w_k=%f' % freqsReal[ind])
        tmp = eng.MATLAB_heat_1d_2('PLvals',cfun,np.complex(1j*freqsReal[ind]),spgrid_ML,0,L1fun,nargout=3)

        # Process the data returned by the Matlab engine
        PLvals[ind] = tmp[0]
        RLBLtmp = tmp[2]
        RLBLpyRE = np.real(RLBLtmp)
        RLBLpyIM = np.imag(RLBLtmp)
        RLBL[ind,] = RLBLpyRE.reshape((N,1)) + 1j*RLBLpyIM.reshape((N,1))

    print('Computation of P_L(iw_k) and RLBL[k] complete!')
    return PLvals, RLBL

