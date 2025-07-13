'''
kinetic inductance detector modeling

Maclean Rouble

maclean.rouble@mail.mcgill.ca
'''





import numpy as np


def calc_Cc(C, f0, Qc, Z0=50):
    return np.sqrt((8*C) / (2*np.pi*f0*Qc*Z0))




#######################
# UTILS
#################

def calc_dphase(Vout):
    x0, y0, R = circle_fit_pratt(Vout.real, Vout.imag)
    phase = np.unwrap(np.arctan2(Vout.imag-y0, Vout.real-x0))
    phase_diff = phase - phase[0]
    return phase_diff

def circle_fit_pratt(x, y):
    """
    Fits a circle to a set of (x, y) points using Pratt's method.
    
    Parameters:
        points (ndarray): Nx2 array of (x, y) points.

    Returns:
        x0 (float): x-coordinate of the circle center
        y0 (float): y-coordinate of the circle center
        R (float): Radius of the circle
    """
    # x = points[:, 0]
    # y = points[:, 1]

    # Construct design matrix
    A = np.column_stack((2*x, 2*y, np.ones_like(x)))
    b = x**2 + y**2

    # Solve the least squares problem: A * p = b
    p = np.linalg.lstsq(A, b, rcond=None)[0]

    x0, y0 = p[0], p[1]  # Circle center
    R = np.sqrt(p[2] + x0**2 + y0**2)  # Circle radius

    return x0, y0, R


def normalize(data, new_min=0, new_max=1):
    """Normalize data to the range [new_min, new_max]."""
    old_min, old_max = np.min(data), np.max(data)
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def square_axes(x):
    '''
    make the axes have the same scale, but allow them to span different ranges, so that the
    resulting plot is square and shows a to-scale representation of the data. (useful for
    IQ circle plots for example)
    
    Parameters:
    -----------
    x : axes object 
    '''
    
    x.set_aspect('equal', adjustable='box')
    x_limits = x.get_xlim()
    y_limits = x.get_ylim()
    max_extent = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    x_mid = (x_limits[0] + x_limits[1]) / 2
    y_mid = (y_limits[0] + y_limits[1]) / 2
    x.set_xlim([x_mid - max_extent / 2, x_mid + max_extent / 2])
    x.set_ylim([y_mid - max_extent / 2, y_mid + max_extent / 2])



def rotate_iq_plane(iqdata, n_thetas=50, use_mean_value=False, make_plots=False, plot_save_dir=None):

    '''
    rotate the iq plane until the stddev is maximized in the 'Q' direction
    '''

    theta_range = np.linspace(0, np.pi, n_thetas)

    imeans = []
    qmeans = []

    istds = []
    qstds = []

    for theta in theta_range:
        theta = 2*np.pi - theta

        #         thisplane = (idata + 1.j*qdata) * np.exp(1.j*theta)
        thisplane = (iqdata) * np.exp(1.j*theta)

        iplane = thisplane.real
        qplane = thisplane.imag
        imeans.append(np.mean(iplane))
        istds.append(np.std(iplane))
        qmeans.append(np.mean(qplane))
        qstds.append(np.std(qplane))

    if make_plots:
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(131)
        bx = fig.add_subplot(132)
        cx = fig.add_subplot(133)

        ax.plot(theta_range, imeans, label='i')
        ax.plot(theta_range, qmeans, label='q')

        bx.plot(theta_range, istds, label='i')
        bx.plot(theta_range, qstds, label='q')

        #ax.axvline(systemic_rotation, color='black', linestyle='--', label='global rotation')
        #bx.axvline(systemic_rotation, color='black', linestyle='--', label='global rotation')

        cx.plot(theta_range, np.asarray(qstds)/np.asarray(istds))
        cx.set_ylabel('Ratio Q/I')

        ax.legend()
        bx.legend()

        ax.set_ylabel('mean')
        bx.set_ylabel('std')
        ax.set_xlabel('reversed IQ rotation')
        bx.set_xlabel('reversed IQ rotation')

        fig.tight_layout()

        if plot_save_dir is not None:
            fig.savefig(os.path.join(plot_save_dir, 'iq_rotation.png'))
            plt.close(fig)

    if use_mean_value:
        ratio = abs(np.asarray(qmeans) / np.asarray(imeans))
    else:
        ratio = np.asarray(qstds) / np.asarray(istds)
    theta_best = 2*np.pi - theta_range[np.argmax(ratio)]
    iqrot = (iqdata) * np.exp(1.j*theta_best)

    return iqrot, theta_best 



##############
# timestreams etc
###################


def calc_rbw(fs, N):
	'''
	resolution bandwidth
	'''

	return fs / N



def make_nqp_timestream_from_Nqp_spectrum(res, frequencies, Nqp_spectrum, rbw, baseline_nqp=None):
    '''
    using the GR noise nqp power spectral density, create a timestream of
    nqp values that corresponds to this spectrum.

    Parameters:
    -----------
    fs : sampling rate

    N : (int) number of samples
    '''

    if baseline_nqp is None:
        baseline_nqp = res.calc_nqp()
    baseline_Nqp = baseline_nqp * res.VL_um3
    
    # rbw = fs / N

    # frequencies = np.fft.fftfreq(N, d=1./fs)
    # frequencies, SN = self.calc_gr_PSD(frange=frequencies)
    SN = Nqp_spectrum

    power_spectrum_N = SN * rbw
    amplitude_spectrum_N = np.asarray(np.sqrt(power_spectrum_N), dtype=complex)

    random_phase_angles = np.random.rand(len(SN)) * 2*np.pi
    random_phases = np.exp(1.j*random_phase_angles)
    amplitude_spectrum_N *= random_phases

    dc_index = abs(frequencies).argmin()
    amplitude_spectrum_N[dc_index] = baseline_Nqp # the average value is the baseline population

    timestream_N = np.fft.ifft(amplitude_spectrum_N).real * len(SN)
    
    return timestream_N / res.VL_um3









