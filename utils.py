'''
kinetic inductance detector modeling

Maclean Rouble

maclean.rouble@mail.mcgill.ca
'''


import numpy as np
from scipy.optimize import curve_fit



def calc_Cc(C, f0, Qc, Z0=50):
    return np.sqrt((2*C) / (2*np.pi*f0*Qc*Z0))




#################
# UTILS
#################

# def calc_dphase(Vout, Vout0):
#     x0, y0, R = circle_fit_pratt(Vout.real, Vout.imag)
#     phase = np.unwrap(np.arctan2(Vout.imag-y0, Vout.real-x0))
#     phase0 = np.arctan2(Vout0.imag-y0, Vout0.real-x0)
#     phase_diff = phase - phase0
#     return phase_diff


def calc_dphase_from_ref_point(Vout, Vout_ref, x0, y0):
#     x0, y0, R = utils.circle_fit_pratt(Vout0.real, Vout0.imag) ### the two ref arrays need the same centre
    phase0 = np.arctan2(Vout_ref.imag-y0, Vout_ref.real-x0)
    dphase = np.arctan2(Vout.imag-y0, Vout.real-x0) - phase0
    
    # phase_diff = phase - phase0
    return dphase

def calc_dphase(Vout, Vout0):
    x0, y0, R = circle_fit_pratt(Vout.real, Vout.imag)
    phase0 = np.arctan2(Vout0.imag-y0, Vout0.real-x0)
    dphase = np.arctan2(Vout.imag-y0, Vout.real-x0) - phase0
    
    # phase_diff = phase - phase0
    return dphase

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


def exp_bin_noise_data(f, psd, nbins=100):
    '''
    Bin a noise PSD, using exponential bin sizes.
    Not statistically rigorous!!
    
    Parameters:
    -----------
    f : list or np.array
        frequency data
    psd : list or np.array
        power spectral density data
        
    Returns:
    --------
    (np.array) binned frequency data
    (np.array) binned PSD data
    '''
    f = np.asarray(f)
    psd = np.asarray(psd)
    
    fbinned = []
    cavg = []
    std = []

    # nbins = 100
    fmin = f[0] + 10e-3 #######
    fmax = f[-1]

    for k in range(nbins-1):
        (fl, fh) = fmin * (fmax / fmin)**(np.asarray([k,k+1])/(nbins - 1.))
        wh = np.where((f>fl)*(f<=fh))[0]

        if len(wh) != 0:
            fbinned.append(np.mean([fl,fh]))
            cavg.append(np.mean(psd[wh]))
            std.append(np.std(psd[wh]))

    fbinned = np.asarray(fbinned)
    cavg = np.asarray(cavg)
    
    return fbinned, cavg



def rotate_iq_plane(iqdata, n_thetas=50, enforce_positive_i=True,
                     use_mean_value=False, make_plots=False, plot_save_dir=None):

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
    
    if enforce_positive_i and np.mean(iqrot.real)<0:
        iqrot *= np.exp(1.j*np.pi)
        theta_best += np.pi

    return iqrot, theta_best 

# def align_carrier_with_readout_I(iqdata):
#     '''
    
#     '''
#     theta_raw = np.arcsin(np.mean(iqdata).imag / abs(np.mean(iqdata))) ### scould this arctan instead?
#     if np.mean(iqdata).real > 0:
#         theta = -theta_raw
        
#     else:
#         theta = theta_raw
        
#     return iqdata * np.exp(1.j*theta), theta
def align_carrier_with_readout_I(iqdata):
    carrier = np.mean(iqdata)

    theta = -np.angle(carrier)
    return iqdata * np.exp(1j * theta), theta


def s21_skewed(f, f0, Qr, Qcre, Qcim, A):
    if abs(Qcre + 1j*Qcim)**2/Qcre < Qr: # prevents negative Qi values
        return np.inf
    else:
        x = (f-f0)/f0
        return abs(A*(1 - Qr/(Qcre+1j*Qcim)/(1+2j*Qr*x)))

def fit_skewed(freq, s21_iq, approxQr=1e4, normalize=True, fr_lim=None, 
                    guess_Qi=None, guess_Qc=None):
    '''
    Apply Pete's skewed resonance model to a given set of s21 vs frequency data.

    NOTE the "sigma" on the fitter uses an arbitrary value but should instead use an estimate
    of system noise.
    
    Parameters
    ----------
    freq : (list or np array) frequency values from a network analysis

    s21_iq : (list or np array) measured values from a network analysis, in i+jq or magnitude format.

    Returns
    -------

    fit_dict : (dict) dictionary of resonance parameters extracted from fit, with uncertainties. If 
        fit fails, returns 'nan'
    
    '''

    bad_fit_flag = False

    param_names = ['fr', 'Qr', 'Qc', 'Qi', 'Qcre', 'Qcim', 'A']
    fit_dict = {}

    freq = np.asarray(freq)
    s21_iq = np.asarray(s21_iq)

    if normalize:
        s21_iq = s21_iq/s21_iq[-1]

    if fr_lim is None:
        fr_lbound = min(freq)
        fr_ubound = max(freq)
        fr_guess = freq[np.argmin(abs(s21_iq))]
    else:
        fr_lbound = np.mean(freq) - fr_lim
        fr_ubound = np.mean(freq) + fr_lim
        fr_guess = np.mean(freq)

    if guess_Qi is not None and guess_Qc is not None:
        approxQr = 1./(1./guess_Qi + 1./guess_Qc)
    else:
        guess_Qc = approxQr
    
    init = [fr_guess, approxQr, guess_Qc, -guess_Qc, abs(s21_iq).mean()]
    bounds = ([fr_lbound, 0, 0, -np.inf, 0], [fr_ubound, np.inf, np.inf, np.inf, np.inf])
    try:
        s21fitp, s21cov = curve_fit(s21_skewed, freq, abs(s21_iq), p0=init, bounds=bounds, sigma=np.ones(len(freq))*90)
        errs = np.sqrt(np.diag(s21cov))
        f0, Qr = s21fitp[0:2]
        Qe = s21fitp[2] + 1j*s21fitp[3]
        Qc = abs(Qe)**2/Qe.real
        Qi = 1./(1/Qr - 1/Qc)
        param_vals = [f0, Qr, Qc, Qi, s21fitp[2], s21fitp[3], s21fitp[4]]
        
        errf0, errQr, errQere, errQeim = errs[0:4] # error in f0, Qr, Qe_re, Qe_im


        # errQc = np.sqrt(errQere**2 + (2 * errQeim * Qe.imag/Qe.real)**2 + errQere*Qe.imag**2/Qe.real**2)
        # errQi = np.sqrt(Qi**4*(errQr**2/Qr**4 + errQc**2/Qc**4))
        a, b = Qe.real, Qe.imag
        if a <= 0 or Qc <= Qr:
            raise ValueError("Fit implies a nonpositive coupling or internal Q")

        # Qc = a + b**2 / a
        grad_Qc = np.array([1 - (b / a)**2, 2 * b / a])
        cov_ab = s21cov[np.ix_([2, 3], [2, 3])]
        errQc = np.sqrt(max(grad_Qc @ cov_ab @ grad_Qc, 0.0))

        # Qi = 1 / (1/Qr - 1/Qc); retain correlations with Qr.
        grad_Qi = Qi**2 * np.array([
            1 / Qr**2,
            -grad_Qc[0] / Qc**2,
            -grad_Qc[1] / Qc**2,
        ])
        cov_Qr_ab = s21cov[np.ix_([1, 2, 3], [1, 2, 3])]
        errQi = np.sqrt(max(grad_Qi @ cov_Qr_ab @ grad_Qi, 0.0))

        errA = errs[-1]
        param_errs = [errf0, errQr, errQc, errQi, errQere, errQeim, errA]

        if np.inf in param_vals:
            print('fit did not converge: found infinity in parameter value.')
            bad_fit_flag = True
        if  np.inf in param_errs:
            print('fit did not converge: found infinity in parameter error.')
            bad_fit_flag = True

        for i in range(len(param_names)):
            fit_dict[param_names[i]] = param_vals[i]
            fit_dict['%s_err'%(param_names[i])] = param_errs[i]


        
    except (RuntimeError, ValueError) as e:
        print(e, '\nfit did not converge')
        bad_fit_flag = True

    if bad_fit_flag:
        for i in range(len(param_names)):
            fit_dict[param_names[i]] = 'nan'
            fit_dict['%s_err'%(param_names[i])] = 'nan'

                
    return fit_dict



##############
# timestreams etc
###################


def calc_rbw(fs, N):
	'''
	resolution bandwidth
	'''

	return fs / N



# def make_nqp_timestream_from_Nqp_spectrum(res, frequencies, Nqp_spectrum, 
#                                             rbw=None, baseline_nqp=None,
#                                             fs=None, N=None):
#     '''
#     using the GR noise nqp power spectral density, create a timestream of
#     nqp values that corresponds to this spectrum.

#     Parameters:
#     -----------
#     fs : sampling rate

#     N : (int) number of samples
#     '''

#     if rbw is None:
#         if fs is None:
#             fs = max(frequencies)*2
#         if N is None:
#             N = len(frequencies)
#         rbw = fs / N

#     if baseline_nqp is None:
#         baseline_nqp = res.calc_nqp()
#     baseline_Nqp = baseline_nqp * res.VL_um3
    
#     # rbw = fs / N

#     # frequencies = np.fft.fftfreq(N, d=1./fs)
#     # frequencies, SN = self.calc_gr_PSD(frange=frequencies)
#     SN = Nqp_spectrum

#     power_spectrum_N = SN * rbw
#     amplitude_spectrum_N = np.asarray(np.sqrt(power_spectrum_N), dtype=complex)

#     random_phase_angles = np.random.rand(len(SN)) * 2*np.pi
#     random_phases = np.exp(1.j*random_phase_angles)
#     amplitude_spectrum_N *= random_phases

#     dc_index = abs(frequencies).argmin()
#     amplitude_spectrum_N[dc_index] = baseline_Nqp # the average value is the baseline population

#     # timestream_N = 2*np.fft.ifft(amplitude_spectrum_N).real * len(SN)
#     timestream_N = 2*np.fft.irfft(amplitude_spectrum_N, n=N) * len(SN)
    
#     return timestream_N / res.VL_um3

def make_nqp_timestream_from_Nqp_spectrum(
    res, frequencies, Nqp_spectrum,
    rbw=None, baseline_nqp=None, fs=None, N=None,
    method='fixed_amplitude'
):
    """Generate a quasiparticle density timestream from a one-sided
    quasiparticle number PSD.
    """
    if N is None:
        N = 2 * (len(frequencies) - 1)

    if rbw is not None:
        if fs is not None and not np.isclose(rbw, fs / N):
            raise ValueError("rbw must equal fs / N")
        fs = rbw * N

    if baseline_nqp is None:
        baseline_nqp = res.calc_nqp()

    number_timestream = make_timestream_from_spectral_density(
        frequencies,
        Nqp_spectrum,
        baseline_value=baseline_nqp * res.VL_um3,
        fs=fs,
        N=N,
        method=method
    )
    return number_timestream / res.VL_um3




# def make_timestream_from_spectral_density(frequencies, spectral_density, baseline_value):
#     '''
#     using the GR noise nqp power spectral density, create a timestream of
#     nqp values that corresponds to this spectrum.

#     Parameters:
#     -----------
#     fs : sampling rate

#     N : (int) number of samples
#     '''

    
#     fs = max(frequencies) * 2
#     N = len(frequencies)
#     rbw = fs / N

#     power_spectrum = spectral_density * rbw
#     amplitude_spectrum = np.asarray(np.sqrt(power_spectrum), dtype=complex)

#     random_phase_angles = np.random.rand(len(power_spectrum)) * 2*np.pi
#     random_phases = np.exp(1.j*random_phase_angles)
#     amplitude_spectrum *= random_phases

#     dc_index = abs(frequencies).argmin()
#     amplitude_spectrum[dc_index] = baseline_value # the average value is the baseline population

#     timestream = np.fft.ifft(amplitude_spectrum).real * len(spectral_density)
    
#     return timestream


def make_timestream_from_spectral_density(
    frequencies, spectral_density, baseline_value,
    fs=None, N=None, method="fixed_amplitude", rng=None
):
    """Generate a real timestream from a one-sided PSD of any shape.

    The PSD must be sampled on the uniform grid produced by
    np.fft.rfftfreq(N, d=1/fs). Its units should be
    (timestream units)^2 / Hz.
    """
    frequencies = np.asarray(frequencies)
    psd = np.asarray(spectral_density)

    if frequencies.ndim != 1 or psd.shape != frequencies.shape:
        raise ValueError(
            "frequencies and spectral_density must be 1D arrays of equal length"
        )
    if len(frequencies) < 2:
        raise ValueError("At least two frequency bins are required")

    # A one-sided FFT of an even-length timestream has N/2 + 1 bins.
    # Callers can supply N explicitly if they want an odd-length timestream.
    if N is None:
        N = 2 * (len(frequencies) - 1)
    if N < 2:
        raise ValueError("N must be at least 2")

    # The final rFFT bin has index N//2. This expression also works when
    # N is odd, in which case the final bin is below the Nyquist frequency.
    if fs is None:
        fs = frequencies[-1] * N / (N // 2)
    if fs <= 0:
        raise ValueError("fs must be positive")

    # An inverse FFT assigns meaning to bins by their position. Check that
    # the supplied frequencies describe those positions before using the PSD.
    expected = np.fft.rfftfreq(N, d=1 / fs)
    if (
        len(frequencies) != len(expected)
        or not np.allclose(frequencies, expected, rtol=1e-10, atol=0)
    ):
        raise ValueError(
            "frequencies must match np.fft.rfftfreq(N, d=1/fs)"
        )

    # The DC coefficient below sets the mean, so the PSD's DC value is unused.
    # This permits, for example, a 1/f PSD that is undefined at 0 Hz.
    if not np.all(np.isfinite(psd[1:])) or np.any(psd[1:] < 0):
        raise ValueError("Non-DC PSD values must be finite and nonnegative")
    if method not in ("fixed_amplitude", "gaussian"):
        raise ValueError(
            "method must be 'fixed_amplitude' or 'gaussian'"
        )

    rng = np.random.default_rng(rng)
    df = fs / N  # Width, in Hz, represented by each Fourier bin.
    spectrum = np.zeros(len(expected), dtype=complex)

    # NumPy's inverse FFT divides by N. Multiplying the desired mean by N
    # makes the mean of the resulting timestream equal baseline_value.
    spectrum[0] = N * baseline_value

    # Interior positive-frequency bins each have a matching negative-frequency
    # bin, which irfft constructs automatically. For even N, exclude the final
    # Nyquist bin because it needs separate, real-valued treatment.
    interior = slice(1, -1 if N % 2 == 0 else None)
    interior_psd = psd[interior]
    count = len(interior_psd)

    if method == "fixed_amplitude":
        # Set each bin's power exactly to its PSD value times df. Random phase
        # changes the waveform without changing that bin's power.
        phases = rng.uniform(0, 2 * np.pi, count)
        spectrum[interior] = (
            N * np.sqrt(interior_psd * df / 2) * np.exp(1j * phases)
        )
    else:
        # Draw independent real and imaginary components. Individual bin
        # powers vary, while their average across realizations matches the PSD.
        # Each component receives half of the complex coefficient's variance.
        scale = N * np.sqrt(interior_psd * df / 4)
        spectrum[interior] = scale * (
            rng.standard_normal(count)
            + 1j * rng.standard_normal(count)
        )

    if N % 2 == 0:
        # The Nyquist bin is its own negative-frequency partner, so its
        # coefficient must be real and uses a different normalization.
        nyquist_scale = N * np.sqrt(psd[-1] * df)
        if method == "fixed_amplitude":
            spectrum[-1] = nyquist_scale * rng.choice((-1.0, 1.0))
        else:
            spectrum[-1] = nyquist_scale * rng.standard_normal()

    # irfft supplies the conjugate negative-frequency bins and returns
    # a real, N-sample timestream using NumPy's standard FFT normalization.
    return np.fft.irfft(spectrum, n=N)







