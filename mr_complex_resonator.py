'''
KID resonator model with superconducting physics.

This module provides the MR_complex_resonator class, which bridges
superconducting physics to the lumped-element circuit model of a kinetic
inductance detector. Given the geometry and material properties of the
superconducting inductor, it computes the Mattis-Bardeen complex conductivity,
derives the surface impedance, and extracts the kinetic inductance and real
impedance needed to instantiate an MR_LEKID circuit model.

On top of the circuit-level calculations provided by MR_LEKID, this class
handles all quasiparticle population physics: thermal and optically-sourced
quasiparticle densities, quasiparticle lifetimes, generation-recombination
noise power spectral densities, and photon noise estimates. Together, these
allow end-to-end simulation of detector response and noise, from absorbed
optical power to output carrier voltage.

The overall modeling approach is described in Chapters 2 and 3 of Rouble
(2025), with the key flow illustrated in Figs. 3.7-3.9.

Link to thesis download: https://escholarship.mcgill.ca/concern/theses/nz806589w?locale=en

Maclean Rouble
maclean.rouble@mail.mcgill.ca
'''

import numpy as np
import copy
from scipy import special
from scipy.optimize import brentq



from mr_lekid import MR_LEKID as MR_LEKID
import utils
import material_properties



h = 6.626e-34
kb = 1.38e-23
# mu0 = 8.85e-12 # F / m
mu0 = 1.257e-6


class MR_complex_resonator(): 
    '''
    KID resonator model connecting superconducting physics to a lumped-element
    circuit.

    In a KID, the sensing element is a superconducting inductor whose kinetic
    inductance and real impedance change when Cooper pairs are broken by
    absorbed photons, altering the quasiparticle population. This class
    computes those changes — starting from the Mattis-Bardeen complex
    conductivity — and propagates them through a lumped-element circuit model
    (MR_LEKID) to predict the measurable output carrier voltage.

    The key chain is (see Fig. 3.8 of Rouble (2025)):

        P_opt  -->  n_qp  -->  sigma (Mattis-Bardeen)
               -->  Z_s   -->  R, L_k  -->  V_out

    On instantiation, the class:
    1. Computes the thermal + optical quasiparticle density n_qp.
    2. Evaluates the Mattis-Bardeen complex conductivity (sigma1, sigma2).
    3. Derives the surface impedance Z_s, and from it the kinetic inductance
       L_k and real impedance R of the inductor.
    4. Instantiates an MR_LEKID with these values (available as self.lekid).

    The resulting MR_LEKID instance (self.lekid) can be used directly to
    compute transfer functions, resonant frequencies, and quality factors.
    The noise and responsivity methods of this class then use the same
    underlying physics to predict the expected detector noise.

    Reference: Rouble (2025), Chapters 2 and 3. Material properties follow
    Gao (2008) (Ph.D. thesis, Caltech).

    Parameters
    ----------
    See __init__ for full parameter descriptions.

    Usage
    -----
    Instantiate a resonator and compute its transfer function::

        res = MR_complex_resonator(T=0.12, material='Al',
                                   length=8.33e-3, width=2e-6, thickness=30e-9,
                                   C=0.5e-12, Cc=5e-15, alpha_k=0.3,
                                   Popt=2.5e-15, Vin=1e-4, input_atten_dB=20)
        fr = res.readout_f
        frange = np.linspace(fr - 500e3, fr + 500e3, 1000)
        Vout = res.lekid.compute_Vout(frange)


    '''
    
    def __init__(self, T=0.12, base_readout_f=1e9, material='Al', VL=540e-18, width=2e-6, thickness=30e-9, 
                 length=None, C=0.5e-12, Cc=0.002e-12, alpha_k=0.5, fix_Lg=None, R_spoiler=0, L_junk=0, 
                 Tc=None, N0=None, tau0=None, Rsheet_N=None, rhoN=None, sigmaN=None,
                 Popt=1e-18, opt_eff=0.5, pb_eff=0.7, nu_opt=150e9, big_sigma_factor=1e-4, nstar=0, 
                 Vin=0.15e-3, input_atten_dB=20, ZLNA=50., GLNA=1, Z0=50.,
                 verbose=False):
        '''
        Parameters
        ----------
        Superconductor geometry
        -----------------------
        width : float
            Inductor linewidth [m]. Default 2e-6.
        thickness : float
            Inductor film thickness [m]. Default 30e-9.
        length : float or None
            Inductor length [m]. If provided, VL is computed from the geometry.
            If None, length is inferred from VL. Default None.
        VL : float
            Active inductor volume [m^3]. Used only if length is None.
            Default 540e-18 (= 9000 um^3).

        Material properties
        -------------------
        material : str
            Superconductor material. Currently 'Al' is supported. Material
            properties (Tc, N0, tau0, sigmaN, etc.) are looked up from
            material_properties.py. Default 'Al'.
        Tc : float or None
            Critical temperature [K]. Overrides the material database value
            if provided. Default None.
        N0 : float or None
            Single-spin density of states [um^-3 J^-1]. Overrides the
            material database value if provided. Default None.
        tau0 : float or None
            Characteristic quasiparticle recombination time [s]. Overrides
            the material database value if provided. Default None.
        Rsheet_N : float or None
            Normal-state sheet resistance [Ohm/sq]. Overrides the material
            database value if provided. Default None.
        rhoN : float or None
            Normal-state resistivity [Ohm m]. Overrides the material database
            value if provided. Default None.
        sigmaN : float or None
            Normal-state conductivity [(Ohm m)^-1]. Overrides the material
            database value if provided. Default None.

        Resonator circuit parameters
        ----------------------------
        C : float
            Resonator shunt capacitance [F]. Default 0.5e-12.
        Cc : float
            Coupling capacitance [F]. Default 0.002e-12.
        alpha_k : float
            Kinetic inductance fraction, L_k / (L_k + L_g). Used to set the
            geometric inductance L_g from the computed L_k. Ignored if
            fix_Lg is provided. Default 0.5.
        fix_Lg : float or None
            If provided, fixes the geometric inductance to this value [H]
            rather than inferring it from alpha_k. Default None.
        R_spoiler : float
            Additional real impedance added to R [Ohm], for modeling excess
            dissipation. Default 0.
        L_junk : float
            Parasitic series inductance passed to MR_LEKID [H]. Default 0.

        Readout circuit parameters
        --------------------------
        Vin : float
            Input carrier amplitude [V] before the last-stage attenuator.
            Default 0.15e-3.
        input_atten_dB : float
            Last-stage attenuator value [dB]. Default 20.
        ZLNA : complex
            LNA input impedance [Ohm]. Default 50.
        GLNA : float
            LNA voltage gain [V/V]. Default 1.
        Z0 : float
            Characteristic transmission line impedance [Ohm]. Default 50.
        base_readout_f : float
            Initial guess for the readout frequency [Hz]. On instantiation,
            this is refined to the computed resonant frequency. Default 1e9.

        Optical loading and quasiparticle parameters
        --------------------------------------------
        T : float
            Operating temperature [K]. Must be less than Tc. Default 0.12.
        Popt : float
            Absorbed pair-breaking optical power [W]. Default 1e-18.
        opt_eff : float
            Optical absorption efficiency, eta_opt <= 1. Default 0.5.
        pb_eff : float
            Pair-breaking efficiency, eta_pb <= 1. Default 0.7.
        nu_opt : float
            Optical photon frequency [Hz], used for photon noise estimates.
            Default 150e9.
        nstar : float
            Effective additional quasiparticle density [um^-3], representing
            a population floor (e.g., from stray pair breaking). Default 0.

        Other
        -----
        big_sigma_factor : float
            Scaling factor for the Goldie-Withington Sigma parameter.
            Default 1e-4.
        verbose : bool
            If True, print circuit parameters on instantiation. Default False.
        '''
        

        self.T = T
        self.readout_f = base_readout_f
        self.Popt = Popt
        self.opt_eff = opt_eff
        self.pb_eff = pb_eff
        self.nu_opt = nu_opt ###
        self.big_sigma_factor = big_sigma_factor #######
        self.material = material

        self.R_spoiler = R_spoiler
        self.L_junk = L_junk
               
        self.width = width
        self.thickness = thickness
        if length is not None:
            self.length = length
            VL = self.width * self.thickness * self.length
        else:
            self.length = VL / (self.width * self.thickness)
        self.VL = VL
        self.VL_um3 = VL*1e18 # in um^3; this is conventionally the units for nqp etc
        
        
        # allow user overrides of various material properties
        overrides = dict()
        overrides.update({
            "Tc": Tc,
            "N0": N0,
            "tau0": tau0,
            "Rsheet_N": Rsheet_N,
            "rhoN": rhoN,
            "sigmaN": sigmaN,
        })

        props = material_properties.resolve_material_properties(self.material, thickness=self.thickness, overrides=overrides)

        # Store base + derived results
        self.Tc = props["Tc"]
        self.N0 = props["N0"]
        self.tau0 = props["tau0"]
        self.Rsheet_N = props["Rsheet_N"]
        self.rhoN = props["rhoN"]
        self.sigmaN = props["sigmaN"]

        if self.T >= self.Tc:
            raise ValueError('Error: cannot set operational temperature equal to transition temperature.')

        self.nstar = nstar
        self.Delta0 = 1.76 * kb * self.Tc
            
        # compute initial guess at resonator dark conductances and circuit values
        nqp = self.calc_nqp(T=T, Popt=self.Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        self.sigma1_initial = self.calc_sigma1(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma2_initial = self.calc_sigma2(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma_initial = self.sigma1_initial - 1.j*self.sigma2_initial
        self.Zs_initial = self.calc_Zs(f=self.readout_f, sigma=self.sigma_initial)
        self.R_initial, self.Lk_initial = self.calc_R_L(f=self.readout_f, Zs=self.Zs_initial)
        self.R_initial += R_spoiler
        
        # generate dark resonator
        self.Z0 = Z0
        self.input_atten_dB = abs(input_atten_dB) ## the code expects a value for ATTENUATION
        self.C = C
        self.Cc = Cc
        self.Vin = Vin
        if fix_Lg is None:
            self.alpha_k = alpha_k
            self.Lg = (self.Lk_initial - self.alpha_k*self.Lk_initial) / self.alpha_k
        else:
            self.Lg = fix_Lg
            self.alpha_k = self.Lk_initial / (self.Lk_initial + self.Lg)
        
        self.lekid_params_initial = dict(R=self.R_initial, Lk=self.Lk_initial, Lg=self.Lg, 
                                        C=self.C, Cc=self.Cc, Vin=self.Vin, 
                                        input_atten_dB=self.input_atten_dB, Z0=self.Z0,
                                        ZLNA=ZLNA, GLNA=GLNA, L_junk=self.L_junk)
        if verbose:
            print('initial parameters:')
            print(self.lekid_params_initial)
        self.lekid_initial = MR_LEKID(**self.lekid_params_initial)
        
        if self.Lk_initial < 0:
            self.readout_f = base_readout_f
            if verbose:
                print('Warning: initial Lk guess is negative.')
        else:
            self.readout_f = self.lekid_initial.compute_fr()
            if verbose:
                print('base readout f: %.4e; readout f now: %.4e'%(base_readout_f, self.readout_f))
        
        # recompute the resonator parameters using the updated readout frequency:
        nqp = self.calc_nqp(T=T, Popt=self.Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        self.sigma1_dark = self.calc_sigma1(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma2_dark = self.calc_sigma2(f=self.readout_f, T=self.T, Popt=self.Popt)
        self.sigma_dark = self.sigma1_dark - 1.j*self.sigma2_dark
        self.Zs_dark = self.calc_Zs(f=self.readout_f, sigma=self.sigma_dark)
        self.R_dark, self.Lk_dark = self.calc_R_L(f=self.readout_f, Zs=self.Zs_dark)
        self.R_dark += R_spoiler
        
        # generate dark resonator
        self.C = C
        self.Cc = Cc
        self.Vin = Vin
        if fix_Lg is None:
            self.alpha_k = alpha_k
            self.Lg = (self.Lk_dark - self.alpha_k*self.Lk_dark) / self.alpha_k
        else:
            self.Lg = fix_Lg
            self.alpha_k = self.Lk_dark / (self.Lk_dark + self.Lg)
        
        self.lekid_params_dark = dict(R=self.R_dark, Lk=self.Lk_dark, Lg=self.Lg, 
                                    C=self.C, Cc=self.Cc, Vin=self.Vin, 
                                    input_atten_dB=self.input_atten_dB, Z0=self.Z0,
                                    ZLNA=ZLNA, GLNA=GLNA, L_junk=self.L_junk)
        self.lekid = MR_LEKID(**self.lekid_params_dark, verbose=verbose)
        self.readout_f = self.lekid.compute_fr()
        if verbose:
            print(self.lekid_params_dark)

    

    def calc_Zs(self, f, sigma, thickness=None):
        '''
        Compute the complex surface impedance from the complex conductivity.

        Uses the thin-film, local-limit expression (Eq. 2.5 of Rouble (2025),
        following Henkels & Kircher 1977):

            Z_s = sqrt(j 2 pi f mu0 / sigma) / tanh(t * sqrt(j 2 pi f mu0 sigma))

        This expression is valid when the film thickness is comparable to or
        smaller than the London penetration depth.

        Parameters
        ----------
        f : float or array
            Frequency at which to evaluate Z_s [Hz].
        sigma : complex or array of complex
            Complex conductivity, sigma1 - j*sigma2 [(Ohm m)^-1].
        thickness : float or None
            Film thickness [m]. Uses self.thickness if None.

        Returns
        -------
        complex or array of complex
            Complex surface impedance Z_s [Ohm/sq].
        '''

        if thickness is None:
            thickness = self.thickness
        root1 = (1.j*2*np.pi*f*mu0)/sigma
        cotharg = thickness * np.sqrt(1.j*2*np.pi*f*mu0*sigma)
        Zs = np.sqrt(root1) * 1./np.tanh(cotharg)
        return Zs

    def calc_Rs_Ls(self, f, Zs):
        '''
        Extract the surface resistance and surface inductance from Z_s.

        Parameters
        ----------
        f : float or array
            Frequency [Hz].
        Zs : complex or array of complex
            Complex surface impedance [Ohm/sq].

        Returns
        -------
        Rs : float or array
            Surface resistance [Ohm/sq].
        Ls : float or array
            Surface inductance [H/sq].
        '''
        Rs = Zs.real 
        Ls = Zs.imag / (2*np.pi*f)
        return Rs, Ls

    def calc_R_L(self, f, Zs):
        '''
        Compute the total inductor resistance R and kinetic inductance L_k
        from the surface impedance and inductor geometry.

        Scales the surface impedance by the length-to-width ratio of the
        inductor (Eqs. 3.1-3.2 of Rouble (2025)):

            L_k = Im{Z_s} / (2 pi f) * (l / w)
            R   = Re{Z_s} * (l / w)

        Note: R_spoiler is added separately in __init__ after this call.

        Parameters
        ----------
        f : float or array
            Frequency [Hz].
        Zs : complex or array of complex
            Complex surface impedance [Ohm/sq].

        Returns
        -------
        R : float or array
            Inductor real impedance [Ohm].
        L : float or array
            Kinetic inductance [H].
        '''
        R = (Zs.real ) * (self.length / self.width) + self.R_spoiler
        L = (Zs.imag / (2*np.pi*f)) * (self.length / self.width)
        return R, L
    


    
    #################################################
    # COMPLEX CONDUCTIVITIES (THERMAL + OPTICAL QP) #
    #################################################

    def zeta(self, f, T):
        '''
        Compute the dimensionless frequency parameter zeta = h*f / (2 k_B T).

        This appears in the Gao (2008) approximations for the Mattis-Bardeen
        conductivity (Eqs. 2.13-2.14 of Rouble (2025)).

        Parameters
        ----------
        f : float or array
            Frequency [Hz].
        T : float
            Temperature [K].

        Returns
        -------
        float or array
            Dimensionless frequency parameter zeta.
        '''
        return h * f / (2 * kb * T)
    

    def calc_sigma1(self, f=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        '''
        Compute the real part of the Mattis-Bardeen complex conductivity.

        Uses the Gao (2008) approximate analytic expression (Eq. 2.13 of
        Rouble (2025)), which treats the quasiparticle density n_qp and
        temperature T as independent variables:

            sigma1 = sigmaN * (2 Delta0 / hf) * (nqp / (N0 sqrt(2 pi kB T Delta0)))
                     * sinh(zeta) * K0(zeta)

        where zeta = hf / (2 kB T) and K0 is the zeroth-order modified Bessel
        function of the second kind.

        sigma1 represents the dissipative (real) response of the
        superconductor. It increases with quasiparticle density and decreases
        at lower temperatures and lower readout frequencies.

        Parameters
        ----------
        f : float or array or None
            Readout frequency [Hz]. Uses self.readout_f if None.
        nqp : float or array or None
            Quasiparticle number density [um^-3]. Computed from T and Popt
            if None.
        T : float or None
            Temperature [K]. Uses self.T if None.
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        pb_eff : float or None
            Pair-breaking efficiency. Uses self.pb_eff if None.
        opt_eff : float or None
            Optical absorption efficiency. Uses self.opt_eff if None.

        Returns
        -------
        float or array
            sigma1 [(Ohm m)^-1].
        '''

        if f is None:
            f = self.readout_f
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if nqp is None:
            nqp = self.calc_nqp(T=T, Popt=Popt, pb_eff=pb_eff, opt_eff=opt_eff)

        zeta = self.zeta(f=f, T=T)
        K0 = special.kn(0, zeta)

        x1 = 2 * self.Delta0/(h*f)
        x2 = nqp / (self.N0 * np.sqrt(2*np.pi*kb*T*self.Delta0))

        return x1 * x2 * np.sinh(zeta) * K0 * self.sigmaN

    
    def calc_sigma2(self, f=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        '''
        Compute the imaginary part of the Mattis-Bardeen complex conductivity.

        Uses the Gao (2008) approximate analytic expression (Eq. 2.14 of
        Rouble (2025)):

            sigma2 = sigmaN * (pi Delta0 / hf)
                     * [1 - (nqp / (2 N0 Delta0)) * (1 + sqrt(2 Delta0 / pi kB T) * e^{-zeta} I0(zeta))]

        where I0 is the zeroth-order modified Bessel function of the first kind.

        sigma2 determines the kinetic inductance of the superconductor via the
        surface impedance. It decreases as n_qp increases (more quasiparticles
        suppress the superfluid density, increasing L_k). At low readout
        frequencies, the reactive response dominates over the dissipative one,
        which makes KIDs most sensitive in this regime (see Fig. 2.2).

        Parameters
        ----------
        f : float or array or None
            Readout frequency [Hz]. Uses self.readout_f if None.
        nqp : float or array or None
            Quasiparticle number density [um^-3]. Computed from T and Popt
            if None.
        T : float or None
            Temperature [K]. Uses self.T if None.
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        pb_eff : float or None
            Pair-breaking efficiency. Uses self.pb_eff if None.
        opt_eff : float or None
            Optical absorption efficiency. Uses self.opt_eff if None.

        Returns
        -------
        float or array
            sigma2 [(Ohm m)^-1].
        '''

        if f is None:
            f = self.readout_f
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if nqp is None:
            nqp = self.calc_nqp(T=T, Popt=Popt, pb_eff=pb_eff, opt_eff=opt_eff)
        Delta0 = self.Delta0
        
        zeta = self.zeta(f=f, T=T)
        I0 = special.iv(0, zeta)

        x1 = np.pi * Delta0 / (h*f)
        x2 = nqp / (2*self.N0*Delta0)
        x3 = np.sqrt(2*Delta0/(np.pi*kb*T)) * np.exp(-zeta) * I0

        return x1 * (1 - x2*(1+x3)) * (self.sigmaN)    
    
    
    def calc_zeroT_sigma2(self, f):
        '''
        Compute the zero-temperature limit of sigma2.

        At T = 0, sigma1 = 0 (no dissipation) and sigma2 approaches a
        frequency-dependent limit set by the gap energy Delta0.

        Parameters
        ----------
        f : float or array
            Frequency [Hz].

        Returns
        -------
        float or array
            sigma2 at T = 0 [(Ohm m)^-1].
        '''

        Delta0 = self.Delta0
        brackets = 1 - (1./16.)*(h*f/Delta0)**2 - (3./1024.) * (h*f / Delta0)**4
        factor = np.pi * Delta0 / (h * f)
        sig2_0 = factor * brackets
        return sig2_0 * self.sigmaN



    ######################################
    # QUASIPARTICLE DENSITY CALCULATIONS #
    ######################################
        
    def calc_nqp(self, T=None, Popt=None, opt_eff=None, pb_eff=None, nstar=None):
        '''
        Compute the steady-state quasiparticle number density.

        Solves the steady-state rate equation (Eq. 2.17 of Rouble (2025))
        balancing thermal generation, optical pair-breaking, and
        recombination:

            n_qp = sqrt(n_th^2 + Gamma_g,opt / R) - n_star

        where R = (2 Delta)^2 / (2 N0 tau0 (kB Tc)^3) is the recombination
        constant, Gamma_g,opt = eta_pb * eta_opt * P_opt / (Delta * V_L)
        is the optical generation rate per unit volume, and n_th is the
        thermal quasiparticle density (Eq. 2.22). The nstar parameter
        represents an additional population floor.

        Parameters
        ----------
        T : float or None
            Temperature [K]. Uses self.T if None.
        Popt : float or None
            Absorbed pair-breaking optical power [W]. Uses self.Popt if None.
        opt_eff : float or None
            Optical absorption efficiency eta_opt. Uses self.opt_eff if None.
        pb_eff : float or None
            Pair-breaking efficiency eta_pb. Uses self.pb_eff if None.

        Returns
        -------
        float or array
            Steady-state quasiparticle number density n_qp [um^-3].
        '''
        
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if nstar is None:
            nstar = self.nstar
            
        Delta = self.calc_Delta_gao(T=T)
        
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        
        nth = self.calc_nqp_th(T=T) 
        rate_thermal = R * nth**2
        rate_optical = 2*pb_eff * opt_eff * Popt / (Delta * self.VL_um3)
                
        return np.sqrt(nth**2 + nstar**2 + rate_optical/R)

    def calc_nqp_th(self, T=None):
        '''
        Compute the thermal equilibrium quasiparticle number density.

        Uses the low-temperature approximation (Eq. 2.22 of Rouble (2025),
        valid for kB T << Delta):

            n_th = 2 N0 sqrt(2 pi kB T Delta) * exp(-Delta / kB T)

        This is the quasiparticle population in the absence of any
        pair-breaking optical load.

        Parameters
        ----------
        T : float or None
            Temperature [K]. Uses self.T if None.

        Returns
        -------
        float
            Thermal quasiparticle number density n_th [um^-3].
        '''

        if T is None:
            T = self.T
        Delta = self.calc_Delta_gao(T)
    
        nqp = 2 * self.N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta / (kb * T))
        return nqp
    
                    

    def calc_tau_qp(self, T=None, Popt=None, opt_eff=None, pb_eff=None, nqp=None):
        '''
        Compute the quasiparticle lifetime tau_qp.

        The quasiparticle lifetime is set by the recombination constant R and
        the total quasiparticle density:

            tau_qp = (N0 (kB Tc)^3) / (2 Delta^2 R nqp)
                   = tau0 N0 (kB Tc)^3 / (2 Delta^2 nqp)

        It sets the rolloff frequency of the GR noise spectrum (~1 / (2 pi tau_qp))
        and decreases as the optical load (and hence n_qp) increases.

        Parameters
        ----------
        T : float or None
            Temperature [K]. Uses self.T if None.
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        opt_eff : float or None
            Optical absorption efficiency. Uses self.opt_eff if None.
        pb_eff : float or None
            Pair-breaking efficiency. Uses self.pb_eff if None.
        nqp : float or None
            Quasiparticle number density [um^-3]. Computed from T and Popt
            if None.

        Returns
        -------
        float
            Quasiparticle lifetime tau_qp [s].
        '''
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if nqp is None:
            nqp = self.calc_nqp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)
            
        Delta = self.calc_Delta_gao(T)
        return (self.tau0 / nqp) * self.N0 * (kb*self.Tc)**3 / (2*Delta**2)



    def calc_gr_PSD(self, frange=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        '''
        Compute the power spectral density of quasiparticle number fluctuations
        from generation-recombination (GR) processes.

        This is the thermally-equilibrated GR noise (Eq. 2.39 of Rouble (2025)),
        which accounts for both generation and recombination fluctuations but
        does not include the additional generation fluctuation due to photon
        shot noise:

            S_N(f) = 4 N_qp tau_qp / (1 + (2 pi f tau_qp)^2)

        The spectrum is white below the rolloff frequency 1/(2 pi tau_qp) and
        falls as 1/f^2 above it. The total number of quasiparticles N_qp
        includes both thermal and optically-sourced contributions.

        Note: this returns NUMBER fluctuations (not number density). To convert
        to number density, divide S_N by V_L^2.

        Parameters
        ----------
        frange : array or None
            Frequencies at which to evaluate the PSD [Hz].
            Default: np.logspace(-2, 5.2, 100).
        nqp : float or None
            Quasiparticle number density [um^-3]. Computed from T and Popt
            if None.
        T : float or None
            Temperature [K]. Uses self.T if None.
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        pb_eff : float or None
            Pair-breaking efficiency. Uses self.pb_eff if None.
        opt_eff : float or None
            Optical absorption efficiency. Uses self.opt_eff if None.

        Returns
        -------
        frange : array
            Frequencies [Hz].
        Sgr : array
            GR noise PSD of quasiparticle number fluctuations [quasiparticles^2 / Hz].
        '''
        
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)
        if nqp is None:
            nqp = self.calc_nqp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)
            
        Nqp = nqp * self.VL_um3
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff) 
        Sgr = 4 * Nqp * tau_qp / (1 + (tau_qp*2*np.pi*frange)**2 )

        return frange, Sgr

    def calc_gr_PSD_thermal_optical(self, frange=None, Popt=None, nu_opt=180e9):
        '''
        Compute the total quasiparticle number PSD including all generation and
        recombination terms.

        Sums the thermal generation (S_g,th), optical generation / photon shot
        noise (S_g,opt), and recombination (S_r) terms (Eqs. 2.35-2.40 of
        Rouble (2025)) and applies the quasiparticle lifetime filter:

            S_N(f) = 2 tau_qp^2 / (1 + (2 pi f tau_qp)^2) * (S_g,th + S_g,opt + S_r)

        This is the quantity used to forecast total on-resonance resonator noise,
        including both thermalized GR noise and the contribution from photon shot
        noise. See Fig. 3.9 and Sec. 5.2.2 of Rouble (2025) for examples of
        this calculation applied to deployed detectors.

        Parameters
        ----------
        frange : array or None
            Frequencies at which to evaluate the PSD [Hz].
            Default: np.logspace(-2, 5.2, 100).
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        nu_opt : float
            Photon frequency used for the photon shot noise term [Hz].
            Default 180e9.

        Returns
        -------
        frange : array
            Frequencies [Hz].
        S_N : array
            Total quasiparticle number PSD [quasiparticles^2 / Hz].
        '''
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)
        if Popt is None:
            Popt = self.Popt

        T=self.T
        opt_eff = self.opt_eff
        pb_eff = self.pb_eff
        T_sky = 20 # K, partially transparent atmosphere
        photon_occupancy = 1. / (np.exp((h * nu_opt)/(kb*T_sky)) - 1)
        
        Delta = self.calc_Delta_gao(T)
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)

        nqp_th = self.calc_nqp_th(T=T)
        nqp = self.calc_nqp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        
        S_r = R * self.VL_um3 * nqp**2
        S_gth = R * self.VL_um3 * nqp_th**2 #
        S_gth = R * self.VL_um3 * nqp**2 # thermalized g spectrum
        S_gopt = (pb_eff / (Delta))**2 * opt_eff * Popt * h * nu_opt * (1 + opt_eff * photon_occupancy)
        prefactor =  2 * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2))
        S_N = (S_gth + S_gopt + S_r) 
        return frange, S_N * prefactor


        
    ##########
    # individual generation and recombination spectra
    #####

    def calc_optical_generation_PSD(self, frange=None, Popt=None, nu_opt=180e9):
        '''
        Compute the PSD of quasiparticle number fluctuations due to photon
        shot noise (optical generation fluctuations).

        This is the contribution from the random arrival times of pair-breaking
        photons (Eq. 2.34 of Rouble (2025)):

            S_g,opt = (eta_pb / Delta)^2 * eta_opt * P_opt * h*nu * (1 + eta_opt * n_bar)

        filtered by the quasiparticle lifetime. Here n_bar is the photon
        occupancy at the sky temperature.

        Parameters
        ----------
        frange : array or None
            Frequencies [Hz]. Default: np.logspace(-2, 5.2, 100).
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        nu_opt : float
            Photon frequency [Hz]. Default 180e9.

        Returns
        -------
        frange : array
            Frequencies [Hz].
        PSD_gopt : array
            Optical generation quasiparticle number PSD [quasiparticles^2 / Hz].
        '''
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)
        if Popt is None:
            Popt = self.Popt

        T=self.T
        opt_eff = self.opt_eff
        pb_eff = self.pb_eff
        T_sky = 20 # K, partially transparent atmosphere
        photon_occupancy = 1. / (np.exp((h * nu_opt)/(kb*T_sky)) - 1)
        Delta = self.calc_Delta_gao(T)
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)

        S_gopt = (pb_eff / (Delta))**2 * opt_eff * Popt * h * nu_opt * (1 + opt_eff * photon_occupancy) 
        PSD_gopt = S_gopt * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2)) * 2
        return frange, PSD_gopt 

    def calc_recombination_PSD(self, frange=None):
        '''
        Compute the PSD of quasiparticle number fluctuations due to
        recombination.

        The recombination process is Poissonian (Eq. 2.35 of Rouble (2025)):

            S_r = 2 R V_L n_qp^2

        filtered by the quasiparticle lifetime. The total quasiparticle
        density n_qp includes both thermal and optically-sourced contributions.

        Parameters
        ----------
        frange : array or None
            Frequencies [Hz]. Default: np.logspace(-2, 5.2, 100).

        Returns
        -------
        frange : array
            Frequencies [Hz].
        S_r : array
            Recombination quasiparticle number PSD [quasiparticles^2 / Hz].
        '''
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)

        T=self.T
        Popt = self.Popt
        opt_eff = self.opt_eff
        pb_eff = self.pb_eff
        Delta = self.calc_Delta_gao(T)
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff) 
        nqp = self.calc_nqp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        
        S_r = 2* R * self.VL_um3 * nqp**2 * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2))
        return frange, S_r

    def calc_thermal_generation_PSD(self, frange=None):
        '''
        Compute the PSD of quasiparticle number fluctuations due to thermal
        pair-breaking.

        The thermal generation fluctuation is (Eq. 2.36 of Rouble (2025)):

            S_g,th = 2 R V_L n_th^2

        filtered by the quasiparticle lifetime. Note that the lifetime uses
        the total quasiparticle density (thermal + optical), since recombination
        depends on all quasiparticles in the system.

        Parameters
        ----------
        frange : array or None
            Frequencies [Hz]. Default: np.logspace(-2, 5.2, 100).

        Returns
        -------
        frange : array
            Frequencies [Hz].
        S_gth : array
            Thermal generation quasiparticle number PSD [quasiparticles^2 / Hz].
        '''
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)

        T = self.T
        Popt = self.Popt
        opt_eff = self.opt_eff
        pb_eff = self.pb_eff
        Delta = self.calc_Delta_gao(T)
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        nqp = self.calc_nqp_th(T=T)
        
        S_gth = 2 * R * self.VL_um3 * nqp**2 * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2))
        return frange, S_gth



        
    ################
    # timestreams of these fluctuations
    #############


    def make_carrier_Vout_timestream_for_nqp_timestream(self, Vin_timestream=None, 
                                                nqp_timestream=None, carrier_freq=None,
                                                  fs=1e5, N=int(1e4)):
        '''
        Convert a timestream of quasiparticle densities to a timestream of
        output carrier voltages.

        For each value of n_qp in the input timestream, this method propagates
        the change through the full physics chain (Fig. 3.8 of Rouble (2025)):

            n_qp  -->  sigma1, sigma2  -->  Z_s  -->  R, L_k  -->  V_out

        instantiating a new MR_LEKID at each timestep with the updated R and
        L_k, and evaluating V_out at the carrier frequency. The carrier frequency
        is held fixed throughout (i.e., the carrier is not updated to track
        the resonance), consistent with typical readout operation.

        This method is the core of the noise forecasting approach illustrated
        in Fig. 3.9 of Rouble (2025): inverse-Fourier transforming the GR noise
        PSD to produce a n_qp timestream, then propagating it to V_out.

        Parameters
        ----------
        nqp_timestream : array
            Timestream of quasiparticle number densities [um^-3].
        Vin_timestream : array or None
            Timestream of input carrier amplitudes [V]. If None, uses a
            constant value of self.Vin for all timesteps.
        carrier_freq : float or None
            Fixed carrier frequency [Hz]. Uses self.readout_f (= fr) if None.
        fs : float
            Sample rate of the timestream [Hz]. Not used internally but
            provided for bookkeeping. Default 1e5.
        N : int
            Number of samples. Not used internally. Default 1e4.

        Returns
        -------
        timestream_Vout : array of complex
            Timestream of complex output carrier voltages [V].
        '''

        if Vin_timestream is None:
            Vin = self.Vin
            Vin_timestream = np.ones(len(nqp_timestream)) * Vin
            
        if carrier_freq is None:
            fr = self.lekid.compute_fr()
            carrier_freq = fr

        
        timestream_s1 = self.calc_sigma1(nqp=nqp_timestream,f=carrier_freq)
        timestream_s2 = self.calc_sigma2(nqp=nqp_timestream,f=carrier_freq)
        timestream_s = timestream_s1 - 1.j*timestream_s2
        timestream_Zs = self.calc_Zs(sigma=timestream_s, f=carrier_freq)
        timestream_R, timestream_Lk = self.calc_R_L(Zs=timestream_Zs, f=carrier_freq)

        timestream_Vout = []
        for r, R in enumerate(timestream_R):
            res_params = copy.deepcopy(self.lekid_params_dark)
            res_params['R'] = R
            res_params['Lk'] = timestream_Lk[r]
            res_params['Vin'] = Vin_timestream[r]
            gr_mkid = MR_LEKID(**res_params)

            timestream_Vout.append(gr_mkid.compute_Vout(carrier_freq))

        timestream_Vout = np.asarray(timestream_Vout)  
        return timestream_Vout


    ###########
    # GENERAL #
    ###########


    def est_photon_noise(self, Popt=None, nu=220e9, opt_eff=None, Tsky=20):
        '''
        Estimate the photon noise-equivalent power (NEP).

        Computes the photon NEP from the random arrival statistics of photons
        (Eq. 2.33 of Rouble (2025)):

            NEP_photon = sqrt(2 eta_opt h nu P_opt (1 + eta_opt n_bar))

        where n_bar is the photon occupancy at the sky temperature Tsky. This
        represents the irreducible noise floor from the photon stream itself.
        A system is photon-noise limited when its total noise is dominated by
        this quantity.

        Parameters
        ----------
        Popt : float or None
            Absorbed optical power [W]. Uses self.Popt if None.
        nu : float
            Photon frequency [Hz]. Default 220e9.
        opt_eff : float or None
            Optical absorption efficiency. Uses self.opt_eff if None.
        Tsky : float
            Sky brightness temperature [K], used to compute the photon
            occupancy n_bar. Default 20.

        Returns
        -------
        float
            Photon NEP [W / sqrt(Hz)].
        '''
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        n_nu = (np.exp(h*nu / (kb * Tsky)) - 1)**(-1)  # photon occupancy
        photon_nep = np.sqrt(2*h*Popt*nu*opt_eff * (1 + opt_eff * n_nu)) # W / rtHz
        return photon_nep
    


    def calc_Delta_gao(self, T=None):
        '''
        Compute the superconducting gap energy at temperature T.

        Uses the Gao (2008) numerical approximation (Eq. 2.9 of Rouble (2025)),
        which is accurate up to T ~ 0.7 Tc:

            Delta(T) = Delta0 * exp(-sqrt(2 pi kB T / Delta0) * exp(-Delta0 / kB T))

        Parameters
        ----------
        T : float or None
            Temperature [K]. Uses self.T if None.

        Returns
        -------
        float
            Gap energy Delta(T) [J].
        '''
        if T is None:
            T = self.T

        x = np.sqrt((2*np.pi*kb*T/self.Delta0) * np.exp(-self.Delta0/(kb*T)))
        return self.Delta0 * np.exp(-x)
        
        # innerexp = np.exp(-self.Delta0 / (kb*T))
        # outerexp = np.exp( (-2*np.pi*kb * T / self.Delta0) * innerexp)
        # return self.Delta0 * outerexp

    def calc_fermi(self, E, T=None):
        '''
        Compute the Fermi-Dirac distribution at energy E and temperature T.

        Parameters
        ----------
        E : float or array
            Energy [J].
        T : float or None
            Temperature [K]. Uses self.T if None.

        Returns
        -------
        float or array
            Fermi-Dirac occupation probability.
        '''
        
        if T is None:
            T = self.T
            
        fermi = 1./( np.exp(E/(kb * T)) + 1 )
        return fermi


    
    ########################################################
    # READOUT POWER ABSORPTION MODELING : QP HEATING MODEL #
    ########################################################

    def calc_power_abs_in_res(self, fc=None, Iin=None, Ires=None, Zres=None):
        '''
        Compute the power dissipated in the resonance at a given frequency.

        params:
        -------
        Ires : current through the resonator
        Zres : impedance of the resonator at a given probe frequency

        '''
        if fc is None:
            fc = self.readout_f
        if Zres is None:
            Zres = self.lekid.total_impedance(fc)
        if Iin is None:
            Iin = self.lekid.calc_Iin(fc=fc, Zres=Zres)
        if Ires is None:
            Ires = self.lekid.calc_Ires(Zres=Zres, Iin=Iin)
            

        power = abs(Ires)**2 * Zres.real
        return power


    def calc_eta(self, Pabs_per_volume):
        '''
        Compute the eta_2Delta parameter in the Goldie expression.
        *** Note that this expects Pabs_per_volume in units of W/um^3 ***
        '''
        eta = -0.03 * np.log(Pabs_per_volume) + 0.384
        return eta    


    def calc_Pabs_times_eta(self, Teff, E=None, Tb=None, true_Pabseta=None):
        '''
        the goldie&withington equation for absorbed power in resonator vs
        effective temperature of the quasiparticle distribution.
        Note that this solves for Pabs * eta(Pabs), in order to put all the Pabs
        terms on the LHS of the expression while the RHS depends only on T and frequency

        params:
        -------
        Teff : effective temperature of the quasiparticle system.
        E : energy (= h * f) for the driven fermi distribution
        Tb : bath temperature (temperature of the metal of the resonator, the
            "bath" for the quasiparticles)
        Tc : transition temperature for the material

        NOTE big_sigma_factor -- a scaling factor, since we don't typically have
            direct measurements of big sigma for our resonators

        '''

        bigsigma = 3.4e10 * 1e-18 # W um^-3 K-1 for Al film
        bigsigma = bigsigma * self.big_sigma_factor
        tau_quotient = 1

        Delta_bath = self.calc_Delta_gao(T=Tb)
        bath_exp = -2*Delta_bath / (kb * Tb)
        bath_term = Tb * np.exp(-2*Delta_bath / (kb * Tb))

        Delta_Teff = self.calc_Delta_gao(T=Teff)
        Teff_exp = -2 * Delta_Teff / (kb * Teff)
        Teff_term = Teff * np.exp(-2 * Delta_Teff / (kb * Teff))

        Pabs_eta = bigsigma * (1. / (1+tau_quotient)) * (Teff_term - bath_term)
        return Pabs_eta - true_Pabseta

    def solve_for_Pabseta_at_T(self, Tguess=None, true_Pabseta=None, fc=None, Tb=None, nguesses=1000, guess_incr=0.0003,
                               pass_flag=False, err_accept=0.1, verbose=False, vv=False, verbose_failonly=False):
        '''
        iteratively evaluate the goldie&withington expression for Pabs * eta(Pabs).
        Starting with a guess for Teff, keep updating the guess Teff until
        the returned Pabs is sufficiently close to the true value (computed elsewhere).
        '''
        if fc is None:
            fc = self.readout_f
        E = fc * h
        if true_Pabseta is None:
            Pabs = self.calc_power_abs_in_res(fc=fc) / self.VL_um3
            true_Pabseta = Pabs * self.calc_eta(Pabs)
            print('checking against true pabseta:', true_Pabseta)
            
        if Tguess is None:
            Tguess = self.T
        if Tb is None:
            Tb = self.T

        accept = False
        
        Tsol, rootresult = brentq(self.calc_Pabs_times_eta, a=Tb, b=self.Tc, args=(E, Tb, true_Pabseta), full_output=True)
        if not rootresult.converged:
            if verbose:
                print('Failed!')
            return Tsol, False
        
        else:
            if verbose:
                print('Passed! Teffs: %f, niters %d'%(Tsol, rootresult.function_calls))
            return Tsol, True
