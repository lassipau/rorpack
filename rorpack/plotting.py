'''Functionality for illustrating the outputs, regulation error and states of the simulated systems, as well as spectral plots.

Copyright (C) 2019 by Lassi Paunonen and the developers of RORPack.
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from matplotlib import animation
from matplotlib import cm
from matplotlib import colors


def plot_spectrum(A, xlim=None, ylim=None):
    '''
    Plots the spectrum of the matrix A in the complex plane.

    Parameters
    ----------
    A : (N, M) array_like
        The square matrix.
    xlim : (,2) array_like, optional
        Limit the view of the spectrum between these two points in x-direction.
    ylim : (,2) array_like, optional
        Limit the view of the spectrum between these two points in y-direction.
    '''
    eigvals = np.linalg.eig(A)[0]
    plt.figure()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.plot(np.real(eigvals), np.imag(eigvals), 'r.')
    plt.grid(True)
    plt.show()

def plot_output(tgrid, output, yref, style='samefig', colorstyle='default'):
    '''
    Plots the reference signal and output as functions of time.

    Parameters
    ----------
    tgrid : (, N) array_like
        The time grid for plotting the signals.
    output : (M, N) array_like
        The output of the controlled system.
    yref : function
        The reference signal.
    style : string, optional
        Style of the plot, either
        'samefig' - all output and reference signal components are 
        plotted in the same figure.
        'subplot' - output and reference signal components are 
        plotted separate subplots in the same figure.
        'separate' - all output and reference signal components are 
        plotted in separate figures
    colorstyle : string, optional
        Matplotlib color scheme of the plot.
    '''
    N = output.shape[0]

    mpl.style.use(colorstyle)

    if style is 'samefig' or N == 1:
        plt.figure(1)
        plt.plot(tgrid, yref(tgrid).T,color='0.5',linewidth=2.0)

        dim_Y = output.shape[0]
        for ind in range(0,dim_Y):
            plt.plot(tgrid, output[ind],label='$y_{' + str(ind+1) +'}(t)$',linewidth=2.0)
        # Print legend if dim_U>1, choose the location of the legend 
        # ('3' = "lower left corner" by default)
        if N>1:
            plt.legend(loc=3)
        plt.title('Output $y(t)$ and the reference $y_{ref}(t)$ (gray)')
        plt.tight_layout()
        plt.grid(True)
    elif style is 'subplot':
        plt.figure(1)
        for i in range(0, N):
            plt.subplot(N, 1, i+1)
            plt.plot(tgrid, yref(tgrid)[i],color='0.5',linewidth=2.0)
            plt.plot(tgrid, output[i],linewidth=2.0)
            plt.title('Output component $y_{%d}(t)$ and $y_{ref, %d}(t)$ (gray)' % (i + 1, i + 1))
            plt.tight_layout()
            plt.grid(True)
    elif style is 'separate':
        for i in range(0, N):
            plt.figure(i + 1)
            plt.plot(tgrid, yref(tgrid)[i],color='0.5',linewidth=2.0)
            plt.plot(tgrid, output[i],linewidth=2.0)
            plt.title('Output component $y_{%d}(t)$ and $y_{ref, %d}(t)$ (gray)' % (i + 1, i + 1))
            plt.tight_layout()
            plt.grid(True)
    plt.show()

def plot_results(tgrid, yref, output, error, colorstyle='default'):
    '''
    [Currently unused, removal considered.]
    Plots the reference signal, output and error as functions of time.

    Parameters
    ----------
    tgrid : (, N) array_like
        The time grid for plotting the signals.
    yref : function
        The reference signal.
    output : (M, N) array_like
        The output of the controlled system.
    error : (M, N) array_like
        The regulation error of the controlled system.
    colorstyle : string, optional
        Matplotlib color scheme of the plot.
    '''
    mpl.style.use(colorstyle)
    N = output.shape[0]

    if N == 1:
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(tgrid, yref(tgrid),linewidth=2.0)
        plt.plot(tgrid, output[0],linewidth=2.0)
        plt.title('Output $y(t)$ in red and the reference $y_{ref}(t)$ in blue')
        plt.subplot(2, 1, 2)
        plt.tight_layout()
        plt.plot(tgrid, error[0],linewidth=2.0)
        plt.grid(True)
        plt.title('Regulation error $y(t) - y_{ref}(t)$')
    else:
        for i in range(0, N):
            plt.figure(i + 1)
            plt.subplot(2, 1, 1)
            plt.plot(tgrid, yref(tgrid)[i],linewidth=2.0)
            plt.plot(tgrid, output[i],linewidth=2.0)
            plt.title('Output $y_{%d}(t)$ in red and the reference $y_{ref, %d}(t)$ in blue' % (i + 1, i + 1))
            plt.subplot(2, 1, 2)
            plt.tight_layout()
            plt.plot(tgrid, error[i],linewidth=2.0)
            plt.grid(True)
            plt.title('Regulation error $y_{%d}(t) - y_{ref, %d}(t)$' % (i + 1, i + 1))
    plt.show()

def plot_error_norm(tgrid, error, colorstyle='default'):
    '''
    Plots the (Euclidean) norm :math:`\\Vert e(t)\\Vert` of the regulation error at the points in 'tgrid'.

    Parameters
    ----------
    tgrid : (, N) array_like
        The time grid for plotting the error norm.
    error : (M, N) array_like
        The regulation error of the controlled system.
    colorstyle : string, optional
        Matplotlib color scheme of the plot.
    '''
    mpl.style.use(colorstyle)

    N = error.shape[0]
    errornorm = np.sqrt(np.sum(np.square(np.abs(error)),axis=0))
    plt.plot(tgrid, errornorm,linewidth=2.0)
    plt.title('Norm of the regulation error $\\Vert y(t) - y_{ref}(t)\\Vert$')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_control(tgrid, control, colorstyle='default'):
    '''
    Plots the control signal :math:`u(t)` at the points in 'tgrid'.

    Parameters
    ----------
    tgrid : (, N) array_like
        The time grid for plotting the error norm.
    control : (M, N) array_like
        The control input.
    colorstyle : string, optional
        Matplotlib color scheme of the plot.
    '''
    mpl.style.use(colorstyle)

    dim_U = control.shape[0]
    for ind in range(0,dim_U):
        plt.plot(tgrid, control[ind],label='$u_{' + str(ind+1) +'}(t)$',linewidth=2.0)
    
    # Print legend if dim_U>1, choose the location of the legend 
    # ('3' = "lower left corner" by default)
    if dim_U>1:
        plt.legend(loc=3)

    plt.title('The control signal $u(t)$')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_1d_surface(tgrid, x, y, colormap=cm.viridis, **options):
    '''
    Plots a 1D surface as a function of time and a single spatial variable. 
    Used for plotting the state of the controlled PDEs with 1-dimensional 
    spatial domains.

    Note: Plots with steep slopes may not look very nice. In this case
    'rcount' and 'ccount' should be changed via options.

    Parameters
    ----------
    tgrid : (, N) array_like
        The time grid for plotting the surface.
    x : (, M) array_like
        The spatial grid for plotting the surface.
    y : (>=M, N) array_like
        The height of the surface as a function of time 't' and 
        spatial variable 'x' 
    options 
        Options passed to the Matplotlib command 'plot_surface'.
    '''

    N = x.shape[0]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Y, X = np.meshgrid(tgrid, x)
    ax.plot_surface(X, Y, y[0:N, :], cmap=colormap, **options)
    plt.xlabel('$\\xi$')
    plt.ylabel('$t$')
    return plt.show()


def animate_1d_results(xxg, y, tgrid, colorstyle='default'):
    '''
    Animates the state of the controlled linear PDE with a 1-dimensional
    spatial domain.

    Parameters
    ----------
    xxg : (, M) array_like
        The spatial grid of the simulation.
    y : (>=M, N) array_like
        The state of the simulated PDE at (x,t)
    tgrid : (, N) array_like
        The time grid of the animation.
    colorstyle : string, optional
        Matplotlib color scheme of the plot.
    '''
    mpl.style.use(colorstyle)
    
    N = xxg.shape[0]
    yy = y[0:N, :]
    if np.amax(np.abs(np.imag(yy))) > 1e-10:
        print('Solution may contain imaginary parts that are ignored in the animation.')
    yy = np.real(yy)
    ymin = np.amin(yy)
    ymax = np.amax(yy)

    fig = plt.figure()
    l, = plt.plot([], [],linewidth=2.0)
    plt.xlim(xxg[0], xxg[-1])
    plt.ylim(ymin, ymax)
    plt.xlabel('$\\xi$')

    def update_line(frame, data, line):
        line.set_data(xxg, data[:, frame])
        plt.title('$t$ = %.2f' % tgrid[frame])
        return line

    ani = animation.FuncAnimation(fig, update_line, np.size(tgrid),
                                  fargs=(yy, l), repeat=False, interval=50)
    return plt.show()


def animate_2d_results(xxg, yyg, zzg, tgrid, colormap=cm.viridis):
    '''
    Animates the state of the controlled linear PDE with a 2-dimensional spatial domain.

    Parameters
    ----------
    xxg : (M, N) array_like
        The x-grid of the simulation.
    yyg : (M, N) array_like
        The y-grid of the simulation.
    zzg : (>=M*N, P) array_like
        The state of the simulated PDE at (x,y,t)
    tgrid : (, P) array_like
        The time grid of the animation.
    colormap : Name of a Matplotlib colormap, optional
        Colormap of the animation, default = cm.viridis
    '''
    N, M = xxg.shape
    zz = zzg[0:N*M, :]
    if np.amax(np.abs(np.imag(zz))) > 1e-10:
        print('Solution may contain imaginary parts that are ignored in the animation.')
    zz = np.real(zz)
    zmin = np.amin(zz)
    zmax = np.amax(zz)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    norm = colors.Normalize(vmin=zmin, vmax=zmax)

    def update(frame):
        ax.clear()
        ax.set_xlim3d([xxg[0, 0], xxg[0, -1]])
        ax.set_ylim3d([yyg[0, 0], yyg[-1, 0]])
        ax.set_zlim3d([zmin, zmax])
        ax.set_title('$t$ = %.2f' % tgrid[frame])
        return ax.plot_surface(xxg, yyg, np.reshape(zz[:, frame], (N, M)),
                               cmap=colormap, norm=norm)
    ani = animation.FuncAnimation(fig, update, np.size(tgrid), repeat=False, interval=50)
    return plt.show()
