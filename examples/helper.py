# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

# Plotting parameters
# ------------------------------------------------------------------------------
params = {
        'text.usetex': True,
        'font.family': 'serif',
         }

plt.rcParams.update(params)

#FIGSIZE1 = (6,5)
#FIGSIZE2 = (6,4)
LW = 2          # line with
LFONTSIZE = 18  # label font size
TFONTSIZE = 15  # tick font size
# ------------------------------------------------------------------------------

def plot_phase_space(data,centroids,labels):
    """Plot the phase space with the snapshots trajectory and the centroids"""
    
    print('Plot phase space')
    print('----------------\n')

    n_cl = centroids.shape[0]

    plt.close()
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111,projection='3d')

    # Data line (grey). For clarity, show only part of the trajectory.
    n_snap = 3000
    ax.plot(
            data[:2*n_snap,0],
            data[:2*n_snap,1],
            data[:2*n_snap,2],
            '-',
            alpha=0.5,
            c='grey',
            zorder=0,
            linewidth=0.5,
            )

    # snapshots with their affiliation. For clarity, show only part of the snapshots
    colors = cm.jet(np.linspace(0,1,n_cl))
    ax.scatter(
            data[::2][:n_snap,0],
            data[::2][:n_snap,1],
            data[::2][:n_snap,2],
            c=colors[labels[::2][:n_snap]],
            label='Data',
            zorder=1,
            alpha=0.5,
            s=5,
            )

    # Centroids
    ax.plot(
            centroids[:,0],
            centroids[:,1],
            centroids[:,2],
            'o',
            color='k',
            zorder=5,
            markersize=7.5,
            )

    # Background and no axes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()

    # Hide grid lines
    ax.grid(False)

    # show the plot
    plt.show()


def plot_time_series(t,x,t_hat,x_hat):
    """Plot the time series of data and CNM"""

    print('Plot time series')
    print('----------------\n')

    # Truncate at the same length
    size = min(t.size,t_hat.size)
    t = t[:size]
    t_hat = t_hat[:size]
    x = x[:size]
    x_hat = x_hat[:size]

    # Show only a specific time range
    t_min,t_max = 45,60
    idx_min, idx_max = np.argmin(abs(t-t_min)), np.argmin(abs(t-t_max))
    t, t_hat, x, x_hat = t[idx_min:idx_max], t_hat[idx_min:idx_max], x[idx_min:idx_max], x_hat[idx_min:idx_max]
    t -= t[0]
    t_hat -= t_hat[0]

    # Plot
    label = ['x','y','z']

    # Initialize figure
    fig1, ax1 = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0},figsize=(6,4.5))
    fig2, ax2 = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0},figsize=(6,4.5))

    for i_dim in range(3):

        ## --> Set plot lims
        #Min = min(UCNM[:,i_dim].min(),UData[:,i_dim].min())
        #Min = (Min + 0.3 * Min) if np.sign(Min) == -1 else (Min - Min * 0.3)
        #print(Min)
        #Max = max(UCNM[:,i_dim].max(),UData[:,i_dim].max())
        #Max = Max + np.sign(Max) * 0.3 * Max
        #yLim = [Min,Max]

        # Start plot Data
        # ----------------------------------------------------------------------
        ax1[i_dim].plot(
                t,
                x[:,i_dim],
                '-',
                c='k',
                lw=LW,
                )

        # Labels
        if i_dim == 2:
            ax1[i_dim].set_xlabel(r'$t$',fontsize=LFONTSIZE)
        ax1[i_dim].set_ylabel(r'${}$'.format(label[i_dim]),fontsize=LFONTSIZE)

        # Lims
        ax1[i_dim].set_xlim([0,t[-1]])

        # --> Ticks
        if i_dim == 2:
            ax1[i_dim].set_yticks([10,30])

        # Start plot CNM
        # ----------------------------------------------------------------------
        ax2[i_dim].plot(
                t_hat,
                x_hat[:,i_dim],
                '-',
                c='r',
                lw=LW,
                )

        # Labels
        if i_dim == 2:
            ax2[i_dim].set_xlabel(r'$t$',fontsize=LFONTSIZE)
        ax2[i_dim].set_ylabel(r'${}$'.format(label[i_dim]),fontsize=LFONTSIZE)

        # Lims
        ax2[i_dim].set_xlim([0,t[-1]])

        # Ticks
        if i_dim == 2:
            ax2[i_dim].set_yticks([10,30])

    for i in range(3):
        # Ticks
        ax1[i].set_xticks([])
        ax2[i].set_xticks([])
        ax1[i].set_yticks([])
        ax2[i].set_yticks([])

        # remove labels and ticks from subplots that are not at the edge of the grid.
        ax1[i].label_outer()
        ax2[i].label_outer()

    # Plot
    fig1.tight_layout()
    fig2.tight_layout()

    #FigName1 = 'xyz-{}-S3.png'.format(PP.Label.replace('.0',''))
    #FigName2 = 'xyz-{}-CNM-S3.png'.format(PP.Label.replace('.0',''))
    #print('--> Saving {}'.format(FigName1))
    #print('--> Saving {}'.format(FigName2))
    #fig1.savefig(FigName1,dpi=500)
    #fig2.savefig(FigName2,dpi=500)
    #c1 = 'convert {F} -trim {F}'.format(F=FigName1)
    #c2 = 'convert {F} -trim {F}'.format(F=FigName2)
    #os.system(c1)
    #os.system(c2)

    plt.show()


def plot_cpd(x,x_hat):
    """Plot the cluster probability vector"""

    print('Plot cluster probability distribution')
    print('-------------------------------------\n')

    # Re-cluster original and CNM data with 10 clusters only for clarity
    from sklearn.cluster import KMeans
    K = 10
    kmeans = KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1)
    kmeans.fit(x)
    labels = kmeans.labels_

    # Predict cluster affiliation of the CNM data
    labels_hat = kmeans.predict(x_hat)

    # Probability distribution
    q = np.bincount(labels).astype(float) / labels.size
    q_hat = np.bincount(labels_hat).astype(float) / labels_hat.size

    # --> Start plot
    # ----------------------------------------------------------------------

    fig = plt.figure(figsize=(6,4))

    x = np.linspace(1,K,K)

    # Plot
    ax = plt.subplot(111)
    ax.bar(
            x-0.1,
            q,
            width=0.2,
            align='center',
            color='k',
            )
    ax.bar(
            x+0.1,
            q_hat,
            width=0.2,
            align='center',
            color='r',
            )

    # Ticks
    ax.set_xticks(np.arange(0,10)+1)
    ax.set_yticks([])
    ax.tick_params(labelsize=TFONTSIZE)


    # Labels
    #TickSize = 14
    ax.set_xlabel(r'$c_k$',fontsize=LFONTSIZE)
    ax.set_ylabel(r'$q$',fontsize=LFONTSIZE)

    # Limits
    max_q = max(q.max(),q_hat.max()) + max(q.max(),q_hat.max()) * 0.1
    ax.set_ylim([0,max_q])

    # Show
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(t,x,t_hat,x_hat):
    """Plot the autocorrelation function"""

    print('Plot autocorrelation function')
    print('-----------------------------\n')

    # Truncate at the same length
    size = min(t.size,t_hat.size)
    t = t[:size]
    t_hat = t_hat[:size]
    x = x[:size]
    x_hat = x_hat[:size]

    # Compute autocorrelation function
    time_blocks = 40
    r = compute_autocorrelation(t,x,time_blocks)
    r_hat = compute_autocorrelation(t_hat,x_hat,time_blocks)
    lags = np.arange(r.size) * (t[-1]-t[0])

    # Plot
    fig = plt.figure(figsize=(6,3.5))
    ax = fig.add_subplot(111)
    
    ax.plot(
            lags,
            r,
            '-',
            c='k',
            lw=LW,
            )
    ax.plot(
            lags,
            r_hat,
            '-',
            c='r',
            lw=LW,
            )
    
    # --> Labels
    ax.set_xlabel(r'$\tau$',fontsize=LFONTSIZE)
    ax.set_ylabel(r'$R$',fontsize=LFONTSIZE)

    # --> Ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # --> Limits
    #ax.set_xlim([-0.1,40])
    #ax.set_ylim([-35,220])

    # --> Plot
    plt.tight_layout()
    #FigName = 'Rxx-{}-S3.png'.format(PP.Label.replace('.0',''))
    #print('--> Saving {}'.format(FigName))
    #plt.savefig(FigName,dpi=500)
    #
    ## --> Trim
    #c = 'convert {F} -trim {F}'.format(F=FigName)
    #os.system(c)
    
    plt.show()

def compute_autocorrelation(t,x,time_blocks: float):
    """Wrapper function to compute the autocorrelation.

    Prepare the data in blocks of length `time_blocks` and calls the appropriate
    function to compute the autocorrelation.
    The autocorrelation is block-averaged.

    Parameters
    ----------

    t: ndarray of shape (n_time steps,)
        The time vector corresponding to the data vector `x`.
    x: ndarray of shape (n_time steps,n_dim)
        Data matrix.
    time_blocks: float
        Time range of the blocks used to compute the autocorrelation block-wise.

    Returns
    -------
    R: ndarray
        Block averaged-autocorrelation of `x`.
    """

    # Split into blocks of time time_blocks
    n_blocks = int(t[-1]/time_blocks)
    x_split = np.array_split(x,n_blocks)
    t_split = np.array_split(t,n_blocks)

    # Truncate the blocks so that they all have the same length
    max_size = min([elt.shape[0] for elt in x_split])
    x_split = [elt[:max_size] for elt in x_split]
    t_split = [elt[:max_size] for elt in t_split]

    # Loop over the blocks
    for i_block,x_block in enumerate(x_split):


        # Remove the mean
        for i_dim in range(x.shape[1]):
            x_block[:,i_dim] -= np.mean(x_block[:,i_dim])

        # fft-based autocorrelation
        r = autocorrelation_fft(x_block)
        tau = np.arange(r.size) * (t[1]-t[0])

        if i_block == 0:
            R = r
        else:
            R += r

    R /= float(n_blocks)
    return R

def autocorrelation_fft(x):
    """Compute the autocorrelation function of x with FFT, IFFT, and 0-padding

    Parameters
    ----------
    x: ndarray
        Data matrix.

    Returns
    -------
    R: ndarray
        Autocorrelation of `x`.
    """

    n = x.shape[0]

    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    # Loop over the dimensions
    for i_dim in range(x.shape[1]):

        # do fft and ifft
        cf=np.fft.fft(x[:,i_dim],fsize)
        sf=cf.conjugate()*cf
        corr=np.fft.ifft(sf).real
        corr=corr/n

        if i_dim == 0:
            r = corr[:n]
        else:
            r += corr[:n]
    return r

def create_lorenz_data():
    """Create the Lorenz data"""

    from scipy.integrate import solve_ivp

    # Lorenz settings
    sigma = 10
    rho   = 28
    beta  = 8/3.
    x0,y0,z0 = (-3,0,31) # Initial conditions
    np.random.seed(0)

    # Lorenz system
    def lorenz(t,q,sigma,rho,beta):
        return [
                sigma * (q[1] - q[0]),
                q[0] * (rho - q[2]) - q[1],
                q[0] * q[1] - beta * q[2],
                ]

    # Time settings
    T = 1000                     # total time
    n_points = T * 60            # number of samples
    t = np.linspace(0,T,n_points)# Time vector
    dt = t[1]-t[0]               # Time step

    # integrate the Lorenz system
    solution = solve_ivp(fun=lambda t, y: lorenz(t, y, sigma,rho,beta), t_span = [0,T], y0 = [x0,y0,z0],t_eval=t)
    data = solution.y.T

    # remove the first 5% to keep only the 'converged' part which is in the ears
    points_to_remove = int(0.05 * n_points)

    return data[points_to_remove:,:], dt
