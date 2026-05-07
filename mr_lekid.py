'''
Lumped-element KID circuit model.

This module provides the MR_LEKID class, which models a kinetic inductance
detector as a lumped-element parallel RLC resonator coupled to a feedline via
a coupling capacitor, embedded within a last-stage attenuator and cryogenic
LNA circuit. It operates purely at the circuit level: given a set of circuit
element values, it computes impedances, transfer functions, resonant
frequencies, and quality factors.

This class is typically instantiated and managed by MR_complex_resonator
(mr_complex_resonator.py), which derives the circuit element values from
superconducting physics. It can also be used directly when circuit element
values are already known.

Reference: Chapter 3 of Rouble (2025), particularly Sec. 3.1-3.2 and
Figs. 3.1-3.5.

Link to thesis download: https://escholarship.mcgill.ca/concern/theses/nz806589w?locale=en

Maclean Rouble
maclean.rouble@mail.mcgill.ca
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import matplotlib as mpl

import utils


class MR_LEKID():
    '''
    Lumped-element circuit model of a KID resonator and surrounding readout
    electronics.

    The resonator is modeled as a parallel RLC circuit (inductance L = Lk + Lg,
    capacitance C, real impedance R) with a series coupling capacitor Cc. This
    resonant branch is treated as the load on the last stage of a pi-type input
    attenuator, in parallel with the LNA input impedance. The topology follows
    Fig. 3.1 of Rouble (2025).

    The transfer function V_out/V_in (Eq. 3.5) models the carrier voltage at the
    LNA output as a function of frequency, given an input carrier amplitude V_in
    before the attenuator. This is directly comparable to an S21 measurement
    from a vector network analyser.

    Parameters
    ----------
    See __init__ for full parameter descriptions.

    Usage
    -----
    Basic frequency sweep::

        lekid = MR_LEKID(Lk=10e-9, Lg=20e-9, C=0.5e-12, Cc=5e-15, R=1e-3,
                         Vin=1e-4, input_atten_dB=20)
        fr = lekid.compute_fr()
        frange = np.linspace(fr - 500e3, fr + 500e3, 1000)
        Vout = lekid.compute_Vout(frange)

    Computing quality factors::

        Qr, Qi, Qc = lekid.compute_Q_values()
    '''
    
    def __init__(self,  C=1e-12, R=1e-6, Cc=5e-15, Lk=1e-9, Lg=None, alpha_k=0.5, L_junk=0,
                 Qi=None, Qc=None, Vin=None, fr_presign2=-1, 
                 system_termination=50., input_atten_dB=20, 
                 Z0=50., ZLNA=complex(50.,0), GLNA=1., 
                 name='LR SERIES',
                LNA_noise_temperature=6, plot_response=False, verbose=False):
        '''
        Parameters
        ----------
        Circuit element parameters
        --------------------------
        C : float
            Resonator shunt capacitance [F]. Default 1e-12.
        R : float
            Resonator series resistance [Ohm]. In a physical KID, this
            is derived from the real part of the surface impedance of the
            superconductor (Eq. 3.2). Default 1e-6.
        Cc : float
            Coupling capacitance [F], in series with the resonator branch.
            Sets the coupling quality factor Qc. Default 5e-15.
        Lk : float
            Kinetic inductance [H]. In a physical KID this is derived from
            the imaginary part of the surface impedance (Eq. 3.1) and changes
            with quasiparticle density. Default 1e-9.
        Lg : float or None
            Geometric inductance [H]. If None, computed from Lk and alpha_k.
            Default None.
        alpha_k : float
            Kinetic inductance fraction, Lk / (Lk + Lg). Used only if Lg
            is None. Default 0.5.
        L_junk : float
            Additional parasitic series inductance [H], if needed to match
            a measured transfer function. Default 0.
        
        Readout circuit parameters
        --------------------------
        Vin : float or None
            Input carrier amplitude [V] before the last-stage attenuator.
            Default 1e-5 if None.
        input_atten_dB : float
            Last-stage attenuator value [dB]. Used to compute the pi-type
            attenuator resistor values. Default 20.
        Z0 : float
            Characteristic transmission line impedance [Ohm]. Default 50.
        ZLNA : complex
            LNA input impedance [Ohm]. Default 50+0j.
        GLNA : float
            LNA voltage gain [V/V]. Default 1 (i.e., gain is not included
            in the transfer function unless explicitly set).
        system_termination : float
            System impedance used for noise calculations [Ohm]. Default 50.
        
        Other parameters
        ----------------
        Qi : float or None
            Internal quality factor. Currently unused in circuit calculations
            (circuit values take precedence). Default None.
        Qc : float or None
            Coupling quality factor. Currently unused in circuit calculations.
            Default None.
        LNA_noise_temperature : float
            LNA noise temperature [K], used to compute LNA noise voltage.
            Default 6.
        name : str
            Label for the resonator instance. Default 'LR SERIES'.
        plot_response : bool
            If True, plot the transfer function on instantiation. Default False.
        verbose : bool
            If True, print resonator parameters on instantiation. Default False.
        '''

        self._init_params = {
            "C": C,
            "R": R,
            "Cc": Cc,
            "Lk": Lk,
            "Lg": Lg,
            "alpha_k": alpha_k,
            "L_junk": L_junk,
            "Qi": Qi,
            "Qc": Qc,
            "Vin": Vin,
            "fr_presign2": fr_presign2,
            "system_termination": system_termination,
            "input_atten_dB": input_atten_dB,
            "ZLNA": ZLNA,
            "Z0": Z0,
            "GLNA": GLNA,
        }

            
        self.Lk = Lk
        if Lg is None:
            self.alpha_k = alpha_k
            self.Lg = (self.Lk - self.alpha_k*self.Lk) / self.alpha_k
        else:
            self.Lg = Lg
            self.alpha_k = self.Lk / (self.Lk + self.Lg)
        self.L = self.Lk + self.Lg
        self.L_junk = L_junk

        self.C = C
        self.R = R
        self.Cc = Cc
        self.name = name
        
            
         # readout params
        self.system_termination = system_termination
        self.input_atten_dB = abs(input_atten_dB)
        self.ZLNA = ZLNA
        self.GLNA = GLNA # LNA gain
        self.Z0 = Z0 # characteristic transmission line impedance
        if Vin is None:
            self.Vin = 1e-5 # arbitrary choice
        else:
            self.Vin = Vin
            
        # noise params
        self.LNA_noise_temperature = LNA_noise_temperature
        self.LNA_noise_vrms_per_rtHz = np.sqrt(1.38e-23 * self.LNA_noise_temperature * 4 * self.system_termination) # over a 1 Hz bw

        self.nonres_flag = False # if the imaginary part of the impedance has no real roots
        
        
        if plot_response:
            self.plot_resonator_response()
            
        if verbose:
            print('Created new resonator, %s, with params:'%(self.name))
            print(self.generate_res_param_string())
        
        
    def parallel_RLC(self, fc, C=None, L=None, R=None):
        '''
        Impedance of the parallel RLC resonator (without the coupling capacitor).

        Computes Z_RLC = [1/Z_C + 1/(Z_L + R)]^{-1} at frequency fc.
        This is the inner resonant branch before adding Cc in series (Eq. 3.3).

        Parameters
        ----------
        fc : float or array
            Carrier frequency [Hz].
        C : float or None
            Shunt capacitance [F]. Uses self.C if None.
        L : float or None
            Total inductance Lk + Lg [H]. Uses self.L if None.
        R : float or None
            Series resistance [Ohm]. Uses self.R if None.

        Returns
        -------
        complex or array of complex
            Impedance of the parallel RLC [Ohm].
        '''
    
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
            
        w = 2*np.pi*fc
        ZC = 1./(1j*w*C)
        ZL = 1j*w*L

        return 1./(1./ZC + 1./(ZL + R))
    
    def total_impedance(self, fc, C=None, L=None, R=None, Cc=None, L_junk=None):
        '''
        Total impedance of the resonator branch, including the series coupling
        capacitor (and optional parasitic series inductance).

        Z_res = Z_RLC + Z_Cc (+ Z_L_junk), as in Eq. 3.4.

        Parameters
        ----------
        fc : float or array
            Carrier frequency [Hz].
        C : float or None
            Shunt capacitance [F]. Uses self.C if None.
        L : float or None
            Total inductance [H]. Uses self.L if None.
        R : float or None
            Series resistance [Ohm]. Uses self.R if None.
        Cc : float or None
            Coupling capacitance [F]. Uses self.Cc if None.
        L_junk : float or None
            Parasitic series inductance [H]. Uses self.L_junk if None.

        Returns
        -------
        complex or array of complex
            Total resonator branch impedance [Ohm].
        '''

        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        if L_junk is None:
            L_junk = self.L_junk
        
        Zres = self.parallel_RLC(fc, L=L, C=C, R=R)
        ZCc = 1./(1j*2*np.pi*fc*Cc)
        ZLjunk = 1j*2*np.pi*L_junk*fc

        return Zres + ZCc + ZLjunk
    
    
    
    def compute_Vout(self, fc, Vin=None, L=None, C=None, R=None, Cc=None, 
                    ZLNA=None, GLNA=None, input_atten_dB=None):
        '''
        Compute the output carrier voltage as a function of frequency.

        Models the resonator as the load of a pi-type last-stage attenuator,
        in parallel with the LNA input impedance. Returns V_out at the LNA
        output, given an input carrier voltage V_in before the attenuator.
        This is the transfer function V_out/V_in * V_in, following Eq. 3.5.

        The shape of |V_out| vs frequency directly resembles an S21 sweep
        measurement from a VNA.

        Parameters
        ----------
        fc : float or array
            Carrier frequency [Hz].
        Vin : float or None
            Input carrier amplitude [V]. Uses self.Vin if None.
        L : float or None
            Total inductance [H]. Uses self.L if None.
        C : float or None
            Shunt capacitance [F]. Uses self.C if None.
        R : float or None
            Series resistance [Ohm]. Uses self.R if None.
        Cc : float or None
            Coupling capacitance [F]. Uses self.Cc if None.
        ZLNA : complex or None
            LNA input impedance [Ohm]. Uses self.ZLNA if None.
        GLNA : float or None
            LNA voltage gain. Uses self.GLNA if None.
        input_atten_dB : float or None
            Last-stage attenuator attenuation value [dB]. Expected to be positive,
            for attenuation in dB. Uses self.input_atten_dB if None.

        Returns
        -------
        complex or array of complex
            Complex output carrier voltage [V] at the LNA output.
        '''
        
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        if Vin is None:
            Vin = self.Vin
        if ZLNA is None:
            ZLNA = self.ZLNA
        if GLNA is None:
            GLNA = self.GLNA
        if input_atten_dB is None:
            input_atten_dB = self.input_atten_dB
            
        r1, r2, r3 = self.get_att_vals(input_atten_dB)
        
        parallel = 1. / ( 1./self.total_impedance(fc, L=L, C=C, R=R, Cc=Cc) + 1./ZLNA )
        
        Vres = Vin * self.ptype(parallel, r1, r2, r3)
        Vout = GLNA * Vres
        return Vout
    

    def ptype(self, rl, r1=61.11, r2=247.5, r3=61.11):
        '''
        Voltage transfer ratio of a pi-type attenuator with load rl.

        Parameters
        ----------
        rl : complex
            Load impedance [Ohm].
        r1, r2, r3 : float
            Pi-type attenuator resistor values [Ohm].

        Returns
        -------
        complex
            V_load / V_in for the attenuator with load rl.
        '''
        req = 1. / (1./r3 + 1./rl)
        VLoverVin = req / (req + r2)
        return VLoverVin


    
    def get_att_vals(self, att, z0=50.):
        '''
        Compute pi-type attenuator resistor values for a given attenuation.
        z0 here is not the resonator line impedance, but the nominal impedance
        this attenuator expects to see. Basically it should always be
        50 ohm, unless you know otherwise.

        Parameters
        ----------
        att : float
            Attenuation [dB].
        z0 : float
            System impedance [Ohm]. Default 50.

        Returns
        -------
        r1, r2, r3 : float
            Resistor values [Ohm] for the pi-type attenuator.
        '''
        att = abs(att)
        r1 = z0 * ((10**(att/20.) +1) / (10**(att/20.) - 1))
        r3 = r1
        r2 = (z0 / 2.) * ((10**(att/10.) - 1) / (10**(att/20.)))

        return r1, r2, r3
    
    def calc_Iin(self, fc, Vin=None, Zres=None):
        '''
        Compute the current entering the parallel network (resonator || LNA)
        at a given frequency.

        Parameters
        ----------
        fc : float
            Carrier frequency [Hz].
        Vin : float or None
            Input carrier amplitude [V]. Uses self.Vin if None.
        Zres : complex or None
            Total resonator branch impedance [Ohm]. Computed if None.

        Returns
        -------
        complex
            Input current to the parallel network [A].
        '''
        if Vin is None:
            Vin = self.Vin
        if Zres is None:
            Zres = self.total_impedance(fc)
        r1, r2, r3 = self.get_att_vals(self.input_atten_dB)
        Zsys = 1. / ( 1./Zres + 1./self.ZLNA )
        Zp = 1. / ( 1./Zsys + 1./r3 )
        I2 = Vin / (r2 + Zp)
        Iin = I2 * ( r3 / (Zsys + r3) )
        return Iin
    
    def calc_Ires(self, fc, Zres=None, Iin=None, Vin=None, ZLNA=50., Z_other=None):
        '''
        Compute the current flowing through the resonator branch using a
        current divider.

        At a given carrier frequency, the input current Iin is divided among
        the attenuator shunt resistance r3, the resonator branch, the LNA,
        and any other specified parallel impedance. This follows Eq. 3.6 in
        Rouble (2025), and is used as an intermediate step in computing the
        current through the inductor (Eq. 3.7) for nonlinearity modeling.

        Parameters
        ----------
        fc : float
            Carrier frequency [Hz].
        Zres : complex or None
            Total resonator branch impedance [Ohm]. Computed if None.
        Iin : complex or None
            Input current to the parallel network [A]. Computed if None.
        Vin : float or None
            Input carrier amplitude [V]. Uses self.Vin if None.
        ZLNA : complex
            LNA input impedance [Ohm]. Default 50.
        Z_other : complex or None
            Additional parallel impedance [Ohm], e.g., a neighbouring
            resonator for crosstalk calculations. Default None.

        Returns
        -------
        complex
            Current through the resonator branch [A].
        '''
        
        if Zres is None:
            Zres = self.total_impedance(fc=fc)
        if Vin is None:
            Vin = self.Vin
        if Iin is None:
            Iin = self.calc_Iin(fc=fc, Zres=Zres, Vin=Vin)

        _, _, r3 = self.get_att_vals(self.input_atten_dB)
            
        if Z_other is not None:
            Zpar = 1./ ( 1./r3 + 1./Zres + 1./ZLNA + 1./Z_other )
        else:
            Zpar = 1./ ( 1./r3 + 1./Zres + 1./ZLNA)
        Ires = Iin * Zpar / Zres
        return Ires
    

    def calc_IL(self, fc, Ires):
        # Zres = new_lekid.total_impedance(carrier_freq)

        ZRLC = self.parallel_RLC(fc)
        ZL = 2.j*np.pi*(self.Lk + self.Lg)*fc
        IL = Ires * ZRLC / (ZL + self.R)

        return IL
    
    
    
    ############
    # L and fr #
    ############
    
    def compute_fr(self, L=None, C=None, R=None, Cc=None, presign2 = 1, verbose=False):
        '''
        Compute the resonant frequency analytically.

        Solves for the frequency at which Im{Z_res} = 0 (the lower of the
        two zero crossings, where |Z_res| is minimised; see Fig. 3.3). The
        solution is the Wolfram Alpha closed-form expression for the
        Cc + (L+R || C) circuit.

        If no real root exists (overdamped resonance), falls back to
        locating the minimum of |V_out| numerically and sets
        self.nonres_flag = True.

        Parameters
        ----------
        L : float or None
            Total inductance [H]. Uses self.L if None.
        C : float or None
            Shunt capacitance [F]. Uses self.C if None.
        R : float or None
            Series resistance [Ohm]. Uses self.R if None.
        Cc : float or None
            Coupling capacitance [F]. Uses self.Cc if None.
        presign2 : int
            Sign selector for the analytic root (+1 or -1). Default 1.
        verbose : bool
            If True, print diagnostic information when no real root is found.

        Returns
        -------
        float
            Resonant frequency [Hz].
        '''
        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        
        D = Cc
        presign = 1 # take only positive roots
        
        # check if we are going to end up with an imaginary solution:
        num3_part2 = 4 * (C**2 * L**2 + C * D * L**2)
        num3_part1 = (C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2
        if num3_part2 > num3_part1:
            # there is no real root to the imaginary part of the impedance
            # instead, look for the local minimum and call this the resonant frequency
            # though, arguable whether or not this is still a resonance
            
            self.nonres_flag = True
            if verbose:
                print('Found unreal solution! Looking for a local minimum instead.\nnum3 root arg: %.2e (num3 part 1: %.2e, num3 part 1: %.2e)'%(num3_part1-num3_part2, num3_part1, num3_part2))
            guess_fr = 1./(np.pi*2 * np.sqrt(L*C))
            guess_Q = 1./(2*np.pi*R*C)
            guess_bw = guess_fr/guess_Q
            
            span = guess_bw*10
            frange = np.linspace(guess_fr-span*2, guess_fr+span/2, 1000)
            test_mag = abs(self.compute_Vout(frange))
            guess_fr = frange[test_mag.argmin()]
            span2 = guess_bw
            frange = np.linspace(guess_fr-span, guess_fr+span, 1000)
            test_mag = abs(self.compute_Vout(frange))
            better_guess_fr = frange[test_mag.argmin()]
            return better_guess_fr

        else:
            numerator3 = -1 * np.sqrt((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2))

        quotient1 = -1 * (C**2 * R**2) / (2 * (C**2 * L**2 + C * D * L**2))
        quotient2 = -1 *(C * D * R**2) / (2 * (C**2 * L**2 + C * D * L**2))
        denom3 = (2 * (C**2 * L**2 + C * D * L**2)) 
        quotient4 = (C * L)/(C**2 * L**2 + C * D * L**2)
        quotient5 = (D * L)/(2 * (C**2 * L**2 + C * D * L**2))

        x = presign * np.sqrt( quotient1 + quotient2 + presign2*(numerator3 / denom3) + quotient4 + quotient5 )

        return x / (2 * np.pi)
    

    def total_impedance_imag(self, fc, C=None, L=None, R=None, Cc=None, L_junk=None):
        '''
        Imaginary part of the total resonator branch impedance.

        Convenience wrapper used internally by compute_fr_numerical().

        Parameters
        ----------
        fc : float or array
            Carrier frequency [Hz].
        C, L, R, Cc, L_junk : float or None
            Circuit element values. Uses instance attributes if None.

        Returns
        -------
        float or array
            Im{Z_res} [Ohm].
        '''

        if C is None:
            C = self.C
        if L is None:
            L = self.L
        if R is None:
            R = self.R
        if Cc is None:
            Cc = self.Cc
        if L_junk is None:
            L_junk = self.L_junk
        
        Zres = self.parallel_RLC(fc, L=L, C=C, R=R)
        ZCc = 1./(1j*2*np.pi*fc*Cc)
        ZLjunk = 1j*2*np.pi*L_junk*fc

        return (Zres + ZCc + ZLjunk).imag
    
    def compute_fr_numerical(self, quantity='IMAG', stepsize_factor=1e-5, nsteps=500, verbose=False, make_plot=False):
        '''
        Compute the resonant frequency numerically using a root-finding method.

        Searches for the frequency at which Im{Z_res} = 0, starting from a
        rough LC estimate and using the Brent method. Useful as a cross-check
        for compute_fr(), or when the analytic solution is unreliable.

        Parameters
        ----------
        quantity : str
            Which quantity to find the zero of. Currently only 'IMAG' is
            supported. Default 'IMAG'.
        stepsize_factor : float
            Step size as a fraction of the initial frequency guess. Default 1e-5.
        nsteps : int
            Maximum number of steps taken to bracket the root. Default 500.
        verbose : bool
            If True, print bracketing information. Default False.
        make_plot : bool
            If True, plot Im{Z} vs frequency with the bracketing bounds and
            the resulting root. Default False.

        Returns
        -------
        float
            Resonant frequency [Hz].

        Raises
        ------
        ValueError
            If no zero crossing is found within nsteps, or if the impedance
            is negative throughout (critically damped).
        '''
    
        if quantity == 'SUM':
            Zfunc = self.total_impedance_component_sum
        else:
            Zfunc = self.total_impedance_imag
    
        guess = 1./(2*np.pi*np.sqrt(self.L*(self.C + self.Cc)))
        stepsize = guess * stepsize_factor
        plot_margin = stepsize * 20
        
        lbound = guess
        for i in range(nsteps):
            lbound -= stepsize
            if Zfunc(lbound) < 0:
                break
        if i == nsteps:
            raise ValueError('Did not find zero crossing in %d steps.'%nsteps)

        ubound = guess-stepsize
        if verbose:
            print('used stepsize: %d Hz\nlbound: %.4f MHz, ubound: %.4f MHz'%(stepsize, 1e-6*lbound, 1e-6*ubound))
            print('Z at bounds: %.2e, %.2e'%(Zfunc(lbound), Zfunc(ubound)))
            
        if lbound < 0 and ubound < 0:
            raise ValueError('Impedance is negative! Resonance critically damped; numerical solution failed.')

        if make_plot:
            frange = np.linspace(lbound-plot_margin, ubound+plot_margin, 100)
            Ztot_imag = Zfunc(frange)
            plt.figure()
            plt.plot(frange, Ztot_imag)
            plt.axvline(lbound, linestyle='--', color='darkgray', label='lower bound')
            plt.axvline(ubound, linestyle='--', color='tab:purple', label='upper bound')
            plt.ylabel('Im(Z)')
            plt.xlabel('Freq. [Hz]')
            plt.ylim(Zfunc(lbound-plot_margin), Zfunc(ubound)/100)
            plt.legend()
                  
        result = brentq(Zfunc, a=lbound, b=ubound)
        if make_plot:
            plt.axvline(result, linestyle='--', color='tab:red', lw=2, label='brentq result:\n$f_r$ = %.4f MHz,\nZ($f_r$) = %.2e'%(result*1e-6, Zfunc(result)))
            plt.legend()
        

        return result


    def compute_Qc(self, Z0=None):
        '''
        Compute the coupling quality factor Qc.

        Qc characterises the rate at which energy is lost from the resonator
        to the feedline. A lower Qc means stronger coupling. The expression
        follows McCarrick's thesis.

        Parameters
        ----------
        Z0 : float or None
            Feedline characteristic impedance [Ohm]. Uses self.Z0 if None.

        Returns
        -------
        float
            Coupling quality factor.
        '''
        if Z0 is None:
            Z0 = self.Z0

        fr = self.compute_fr()
        Qc = (2 * self.C) / (self.Cc**2 * (2 * np.pi * fr * Z0) )
        
        return Qc

    def compute_Qi(self):
        '''
        Compute the internal (intrinsic) quality factor Qi.

        Qi characterises the rate at which energy is dissipated within the
        resonator itself (dominated by the real impedance R of the
        superconductor).

        Returns
        -------
        float
            Internal quality factor.
        '''
        fr = self.compute_fr()
        L = self.Lk + self.Lg
        Qi = np.pi * fr *2 * L / self.R
        return Qi

    def compute_Qr(self):
        '''
        Compute the total (loaded) quality factor Qr.

        1/Qr = 1/Qi + 1/Qc.

        Returns
        -------
        float
            Total quality factor.
        '''
        Qi = self.compute_Qi()
        Qc = self.compute_Qc()
        Qr = 1./(1./Qi + 1./Qc)
        return Qr

    def compute_Q_values(self, Z0=None):
        '''
        Compute Qr, Qi, and Qc together.

        Parameters
        ----------
        Z0 : float or None
            Feedline characteristic impedance [Ohm]. Uses self.Z0 if None.

        Returns
        -------
        Qr, Qi, Qc : float
            Total, internal, and coupling quality factors.
        '''

        Qr = self.compute_Qr()
        Qi = self.compute_Qi()
        Qc = self.compute_Qc(Z0=Z0)
        return Qr, Qi, Qc

    def fit_for_Q_values(self, span=300e3, npts=1000):
        '''
        Estimate Q values by fitting a skewed Lorentzian to the computed
        transfer function.

        Generates a synthetic S21 sweep over a frequency span centred on fr,
        then fits it using the asymmetric Lorentzian fitter in utils.

        Parameters
        ----------
        span : float
            Half-span of the frequency range around fr [Hz]. Default 300e3.
        npts : int
            Number of frequency points. Default 1000.

        Returns
        -------
        Qr, Qi, Qc : float
            Quality factors from the fit.
        '''

        fr = self.compute_fr()
        frange = np.linspace(fr-span, fr+span, npts)
        Vout = self.compute_Vout(frange)
        guess_Qr, guess_Qi, guess_Qc = self.compute_Q_values()
        fit_dict = utils.fit_skewed(frange, Vout, guess_Qc=guess_Qc, guess_Qi=guess_Qi)

        Qr = fit_dict['Qr']
        Qi = fit_dict['Qi']
        Qc = fit_dict['Qc']
        return Qr, Qi, Qc
        
        
    

    #####
    # extras
    #####
    
    def generate_res_param_string(self):
        '''Return a formatted string summarising the key resonator parameters.'''
        res_param_string = 'Lk=%.2e H, Lg=%.2e H, C=%.2e F, Cc=%.2e F, R=%.2e ohm'%(self.Lk, self.Lg, self.C, self.Cc, self.R)
        return res_param_string
    
    
    def plot_resonator_response(self, span=500e3, npts=1000):
        '''
        Plot the resonator transfer function (|V_out|, V_I, V_Q) vs frequency.

        Parameters
        ----------
        span : float
            Half-span of the frequency range around fr [Hz]. Default 500e3.
        npts : int
            Number of frequency points. Default 1000.
        '''

        fr = self.compute_fr()
        frange = np.linspace(fr-int(span), fr+int(span), npts)
        plotfrange = 1e-3*(frange - fr)

        Vout = self.compute_Vout(frange)

        Iout = Vout.real
        Qout = Vout.imag

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)


        ax.plot(plotfrange, Iout, '--', color='royalblue', label='$V_I$')
        ax.plot(plotfrange, Qout, ':', color='royalblue', label='$V_Q$')
        ax.plot(plotfrange, abs(Vout), '-', color='royalblue', lw=2, label='|$V_{out}$|')

        ax.set_title('Resonator frequency response\n$f_r$ = %d MHz; $V_{in}$=%.1f $\\mu V$'%(1e-6*fr, 1e6*self.Vin.real))

        ax.set_ylabel('V$_{out}$')
        ax.set_xlabel('Freq. offset from f$_r$ [kHz]')

        ax.legend(loc='lower right')

        fig.tight_layout()
