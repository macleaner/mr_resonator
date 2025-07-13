'''
kinetic inductance detector modeling

Maclean Rouble

maclean.rouble@mail.mcgill.ca
'''

import numpy as np
import copy
from scipy import special
from scipy.optimize import brentq



from mr_lekid import MR_LEKID as MR_LEKID
import utils


h = 6.626e-34
kb = 1.38e-23
mu0 = 8.85e-12 # F / m


class MR_complex_resonator(): 
    
    def __init__(self, T=0.12, base_readout_f=1e9, material='Al', VL=540e-18, width=2e-6, thickness=30e-9, 
                 length=None, C=0.5e-12, Cc=0.01e-12, alpha_k=0.5, fix_Lg=None, R_spoiler=0, L_junk=0, Tc=None, 
                 Popt=1e-18, opt_eff=0.5, pb_eff=0.7, nu_opt=150e9, big_sigma_factor=1e-4, nstar=0, sigmaN=1./(4*20e-9),
                 Vin=0.15e-3, input_atten_dB=20,
                 verbose=False):
        self.T = T
        self.readout_f = base_readout_f
        self.Popt = Popt
        self.opt_eff = opt_eff
        self.pb_eff = pb_eff
        self.nu_opt = nu_opt ###
        self.big_sigma_factor = big_sigma_factor #######
        if material != 'Al':
            raise ValueError('You must choose Al for aluminum. More options may be added in future.')
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
        
        
        self.sigmaN = sigmaN
    
        if Tc is None:
            self.Tc = 1.2 # for Al
        else:
            self.Tc = Tc
        if self.T >= self.Tc:
            raise ValueError('Error: cannot set operational temperature equal to transition temperature.')
        if material == 'Al': # this is your only choice 
            self.N0 = 1.72e10 * 1.602e19 # for Al, um^-3 eV^-1 --> um^-3 J^-1
            self.tau0 = 438e-9 # s; from de Visser thesis for aluminum (characteristic electron-phonon interaction time)
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
        self.input_atten_dB = input_atten_dB
        self.C = C
        self.Cc = Cc
        self.Vin = Vin
        if fix_Lg is None:
            self.alpha_k = alpha_k
            self.Lg = (self.Lk_initial - self.alpha_k*self.Lk_initial) / self.alpha_k
        else:
            self.Lg = fix_Lg
            self.alpha_k = self.Lk_initial / (self.Lk_initial + self.Lg)
        
        self.lekid_params_initial = dict(R=self.R_initial, Lk=self.Lk_initial, Lg=self.Lg, C=self.C, Cc=self.Cc, Vin=self.Vin, input_atten_dB=self.input_atten_dB, L_junk=self.L_junk)
        if verbose:
            print('initial parameters:')
            print(self.lekid_params_initial)
        self.lekid_initial = MR_LEKID(**self.lekid_params_initial)
        
        if self.Lk_initial < 0:
            self.readout_f = base_readout_f
            if verbose:
                print('Warning: initial Lk guess is negative.')
        else:
#             self.readout_f = 1./np.sqrt(2*np.pi*(self.Lk_dark + self.Lg) * self.C)
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
        
        self.lekid_params_dark = dict(R=self.R_dark, Lk=self.Lk_dark, Lg=self.Lg, C=self.C, Cc=self.Cc, Vin=self.Vin, input_atten_dB=self.input_atten_dB, L_junk=self.L_junk)
        self.lekid = MR_LEKID(**self.lekid_params_dark, verbose=verbose)
        self.readout_f = self.lekid.compute_fr()
        if verbose:
            print(self.lekid_params_dark)

    

    def calc_Zs(self, f, sigma, thickness=None):#, sigma2=None):
        '''
        compute complex surface impedance from complex conductance
        expression from Henkels&Kircher 1977, via de Visser thesis eq 2.20

        params:
        -------
        f : frequency
        sigma : complex conductance = sigma1 - 1.j*sigma2

        '''

        if thickness is None:
            thickness = self.thickness
#         print(thickness)
        root1 = (1.j*2*np.pi*f*mu0)/sigma
        cotharg = thickness * np.sqrt(1.j*2*np.pi*f*mu0*sigma)
        Zs = np.sqrt(root1) * 1./np.tanh(cotharg)
        return Zs

    def calc_Rs_Ls(self, f, Zs):
        '''
        from the complex surface impedance, compute the SURFACE resistance and indutance
        '''
        Rs = Zs.real 
        Ls = Zs.imag / (2*np.pi*f)
        return Rs, Ls

    def calc_R_L(self, f, Zs):
        '''
        from the complex surface impedance, compute the total resistance and kinetic inductance
        '''
        R = (Zs.real ) * (self.length / self.width) + self.R_spoiler
        L = (Zs.imag / (2*np.pi*f)) * (self.length / self.width)
        return R, L
    


    
    #################################################
    # COMPLEX CONDUCTIVITIES (THERMAL + OPTICAL QP) #
    #################################################

    def zeta(self, f, T):
        return h * f / (2 * kb * T)
    

    def calc_sigma1(self, f=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        '''
        complex conductance as function of quasiparticle density AND temperature (ie, to account for
        the presence of optically sourced quasiparticles as well as thermal equilibrium populations.
        From Gao, eq 2.96-2.97
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
        complex conductance as function of quasiparticle density AND temperature (ie, to account for
        the presence of optically sourced quasiparticles as well as thermal equilibrium populations.
        From Gao, eq 2.96-2.97
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
        zero temperature value for imaginary component of conductance
        real part of complex conductance is 0 at T=0

        params
        ------
        f : frequency

        '''

        Delta0 = self.Delta0
        brackets = 1 - (1./16.)*(h*f/Delta0)**2 - (3./1024.) * (h*f / Delta0)**4
        factor = np.pi * Delta0 / (h * f)
        sig2_0 = factor * brackets
        return sig2_0 * self.sigmaN



    ######################################
    # QUASIPARTICLE DENSITY CALCULATIONS #
    ######################################
        
    def calc_nqp(self, T=None, Popt=None, opt_eff=None, pb_eff=None):
        
        if T is None:
            T = self.T
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        if pb_eff is None:
            pb_eff = self.pb_eff
            
        Delta = self.calc_Delta_gao(T=T)
        
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        
        nth = self.calc_nqp_th(T=T) + self.nstar
        rate_thermal = R * nth**2
        rate_optical = pb_eff * opt_eff * Popt / (Delta * self.VL_um3)
                
        return np.sqrt(nth**2 + rate_optical/R) - self.nstar

    def calc_nqp_th(self, T=None):
        '''
        compute quasiparticle number density at temperature T
        '''

        if T is None:
            T = self.T
        Delta = self.calc_Delta_gao(T)
    
        nqp = 2 * self.N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta / (kb * T))
        return nqp
    
                    

    def calc_tau_qp(self, T=None, Popt=None, opt_eff=None, pb_eff=None, nqp=None):
        '''
        compute the quasiparticle lifetime 
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
        #nqp = self.calc_nqp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)
        return (self.tau0 / nqp) * self.N0 * (kb*self.Tc)**3 / (2*Delta**2)



    def calc_gr_PSD(self, frange=None, nqp=None, T=None, Popt=None, pb_eff=None, opt_eff=None):
        '''
        power spectral density of quasiparticle number fluctuations (GR noise)
        does not include the generation fluctuation due to photon noise

        also note that this is NUMBER fluctuation, not number density 

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
        resoantor noise PSD when qp are coming from thermal and optical processes
        includes generation and recombination, plus generation from photon shot noise
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
        S_gth = R * self.VL_um3 * nqp_th**2
        S_gopt = (pb_eff / (Delta))**2 * opt_eff * Popt * h * nu_opt * (1 + opt_eff * photon_occupancy)
        # print('GR total')
        # print('%.1e'%S_r, '%.1e'%S_gth, '%.1e'%S_gopt)
        prefactor =  2 * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2))
        S_N = (S_gth + S_gopt + S_r) 
        # print('%.1e'%S_N[0], '%.1e\n'%tau_qp)
        return frange, S_N * prefactor#, S_N, S_r, S_gth, S_gopt, prefactor

        

    ##########
    # individual generation and recomination spectra
    #####

    def calc_optical_generation_PSD(self, frange=None, Popt=None, nu_opt=180e9):
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)
        if Popt is None:
            Popt = self.Popt

        T=self.T
        # Popt = self.Popt
        opt_eff = self.opt_eff
        pb_eff = self.pb_eff
        T_sky = 20 # K, partially transparent atmosphere
        photon_occupancy = 1. / (np.exp((h * nu_opt)/(kb*T_sky)) - 1)
        Delta = self.calc_Delta_gao(T)
        # R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff)  # I guess this should still be from the TOTAL nqp present...?

        # S_gopt = (pb_eff / (Delta * self.VL_um3))**2 * opt_eff * Popt * h * nu_opt * (1 + opt_eff * photon_occupancy) 
        S_gopt = (pb_eff / (Delta))**2 * opt_eff * Popt * h * nu_opt * (1 + opt_eff * photon_occupancy) 
        PSD_gopt = S_gopt * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2)) * 2
        # print('S gopt')
        # print('%.1e'%PSD_gopt[0], '%.1e'%S_gopt, '%.1e\n'%tau_qp)
        return frange, PSD_gopt 

    def calc_recombination_PSD(self, frange=None):
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
        if frange is None:
            frange = np.logspace(-2, 5.2, 100)

        T = self.T
        Popt = self.Popt
        opt_eff = self.opt_eff
        pb_eff = self.pb_eff
        Delta = self.calc_Delta_gao(T)
        R = (2 * Delta)**2 / (2*self.N0 * self.tau0 * (kb * self.Tc)**3)
        tau_qp = self.calc_tau_qp(T=T, Popt=Popt, opt_eff=opt_eff, pb_eff=pb_eff) # I guess this should also be from the total nqp, including optically sourced...?
        nqp = self.calc_nqp_th(T=T)
        
        S_gth = 2 * R * self.VL_um3 * nqp**2 * (tau_qp**2 / (1 + (tau_qp * 2 * np.pi * frange)**2))
        return frange, S_gth



        
    ################
    # timestreams of these fluctuations
    #############


    def make_carrier_Vout_timestream_for_nqp_timestream(self, Vin_timestream=None, nqp_timestream=None, carrier_freq=None,
                                                  fs=1e5, N=int(1e4)):


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
        # print('starting lekid params:')
        # print(res.lekid_params_dark)
        for r, R in enumerate(timestream_R):
            # print('lekid params now:')
            # print(res.lekid_params_dark)
            res_params = copy.deepcopy(self.lekid_params_dark)
            res_params['R'] = R
            res_params['Lk'] = timestream_Lk[r]
            res_params['Vin'] = Vin_timestream[r]
            gr_mkid = MR_LEKID(**res_params)
            # print('GR mkid params now:')
            # print(gr_mkid.R, gr_mkid.Lk, gr_mkid.Lg)

            timestream_Vout.append(gr_mkid.compute_Vout(carrier_freq))

        timestream_Vout = np.asarray(timestream_Vout)  
        return timestream_Vout





    ###########
    # GENERAL #
    ###########


    def est_photon_noise(self, Popt=None, nu=220e9, opt_eff=None):
        if Popt is None:
            Popt = self.Popt
        if opt_eff is None:
            opt_eff = self.opt_eff
        # h = 6.626e-34 # Js
        # kb = 1.38e-23 # m^2 kg / (s^2 K)
        Tsky = 20 # K, approx black body atmosphere
        n_nu = (np.exp(h*nu / (kb * Tsky)) - 1)**(-1)  # photon occupancy
        photon_nep = np.sqrt(2*h*Popt*nu*opt_eff * (1 + opt_eff * n_nu)) # W / rtHz
        return photon_nep
    


    def calc_Delta_gao(self, T=None):
        '''
        compute the gap energy at temperature T, using the approximation in
        Gao's thesis. This is representative of the full solution up to
        temperatures around 0.7Tc.

        params:
        -------
        T : temperature in K

        '''
        if T is None:
            T = self.T
        
        innerexp = np.exp(-self.Delta0 / (kb*T))
        outerexp = np.exp( (-2*np.pi*kb * T / self.Delta0) * innerexp)
        return self.Delta0 * outerexp

    def calc_fermi(self, E, T=None):
        '''
        fermi-dirac distribution at E = h*f and temperature T in K
        '''
        
        if T is None:
            T = self.T
            
        fermi = 1./( np.exp(E/(kb * T)) + 1 )
        return fermi


    
    ########################################################
    # READOUT POWER ABSORPTION MODELING : QP HEATING MODEL #
    ########################################################

#     def calc_Ires(self, Zres, Iin, carrier_freq=None, ZLNA=50.):
#         '''
#         use a current divider to estimate the current flowing through the resonator.
#         technically this should be a three-way divider between the last-stage attenuator,
#         the resonator, and the LNA input impedance, but this seems fine as an approximation.

#         params:
#         -------
#         Zres : impedance of the resonator at the probe frequency
#         Iin : input current. This is the current entering the divider, analogous to the
#             "fixed" Vin. Typical values are in the range of 1 - 100 nA.
#         '''
#         if carrier_freq is None:
#             carrier_freq = self.readout_f
            
#         Ires = self.lekid.calc_Ires(fc=carrier_freq, Zres=Zres, Iin=Iin, ZLNA=ZLNA)
# #         Ires = Iin * ( (Zres + ZLNA) / Zres)
#         return Ires

    def calc_power_abs_in_res(self, fc=None, Iin=None, Ires=None, Zres=None):
        '''
        compute the power dissipated(/absorbed) in the resonance at a given frequency

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
        compute the eta_2Delta parameter in the Goldie expression.
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




    

    

