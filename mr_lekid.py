'''
kinetic inductance detector modeling

Maclean Rouble

maclean.rouble@mail.mcgill.ca
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import matplotlib as mpl


class MR_LEKID():
    
    def __init__(self,  C=1e-12, R=1e-6, Cc=5e-15, Lk=1e-9, Lg=None, alpha_k=0.5, L_junk=0,
                 Qi=None, Qc=None, Vin=None, fr_presign2=-1, 
                 system_termination=50., input_atten_dB=20, ZLNA=complex(50.,0), GLNA=1., name='LR SERIES',
                LNA_noise_temperature=6, plot_response=False, verbose=False):
        
            
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
        self.input_atten_dB = input_atten_dB
        self.ZLNA = ZLNA
        self.GLNA = GLNA # LNA gain
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
            #print('Created new resonator, %s, with params:\nLk=%.2e H, Lg=%.2e H, C=%.2e F, Cc=%.2e F, R=%.2e ohm.'%(self.name, self.Lk, self.Lg, self.C, self.Cc, self.R))
            print(self.generate_res_param_string())
        
        
    def parallel_RLC(self, fc, C=None, L=None, R=None):
        # where fc is the carrier frequency
        # Compute the impedance of the parallel RLC only
    
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
        # where fc is the carrier frequency
        # Compute total device impedance at fc,
        # including the coupling capacitor in series with
        # the parallel RLC resonance

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
    
    
    
    def compute_Vout(self, fc, Vin=None, L=None, C=None, R=None, Cc=None, ZLNA=None, GLNA=None, input_atten_dB=None):
        # Model the resonator as a voltage divider between it in parallel
        # with a complex input impedance of the LNA and an input 50 ohm attenuator,
        # return the voltage of the carrier across the resonator at a given frequency
        # as Vout, given an input carrier voltage at that frequency before the attenuator
        
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
        attenuator attenuation with a given load rl
        '''
        req = 1. / (1./r3 + 1./rl)
        VLoverVin = req / (req + r2)
        return VLoverVin


    
    def get_att_vals(self, att, z0=50.):
        '''
        ptype attenuator component calculator
        for nominal 50 ohm values
        '''
        
        r1 = z0 * ((10**(att/20.) +1) / (10**(att/20.) - 1))
        r3 = r1
        r2 = (z0 / 2.) * ((10**(att/10.) - 1) / (10**(att/20.)))

        return r1, r2, r3
    
    def calc_Iin(self, fc, Vin=None, Zres=None):
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
        use a current divider to estimate the current flowing through the resonator.
        technically this should be a three-way divider between the last-stage attenuator,
        the resonator, and the LNA input impedance, but this seems fine as an approximation.

        params:
        -------
        fc: carrier frequency [Hz]
        Zres : impedance of the resonator at the probe frequency
        Iin : input current. This is the current entering the divider, analogous to the
            "fixed" Vin. Typical values are in the range of 1 - 100 nA.
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
    
    
    
    
    ############
    # L and fr #
    ############
    
    def compute_fr(self, L=None, C=None, R=None, Cc=None, presign2 = 1, verbose=False):
        '''
        wolfram alpha solution for the Cc+(L+R || C)
        '''
        # x = -sqrt(-(C^2 R^2)/(2 (C^2 L^2 + C D L^2)) - (C D R^2)/(2 (C^2 L^2 + C D L^2)) - (i sqrt((i C^2 R^2 + i C D R^2 - 2 i C L - i D L)^2 - 4 i (i C^2 L^2 + i C D L^2)))/(2 (C^2 L^2 + C D L^2)) + (C L)/(C^2 L^2 + C D L^2) + (D L)/(2 (C^2 L^2 + C D L^2)))
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
            
#             numerator3 = -1 * np.sqrt(abs((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2)))
        else:
            numerator3 = -1 * np.sqrt((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2))

        quotient1 = -1 * (C**2 * R**2) / (2 * (C**2 * L**2 + C * D * L**2))
        quotient2 = -1 *(C * D * R**2) / (2 * (C**2 * L**2 + C * D * L**2))
#         numerator3 = -1 * np.sqrt((C**2 * R**2 + C * D * R**2 - 2 * C * L - D * L)**2 - 4 * (C**2 * L**2 + C * D * L**2))
        denom3 = (2 * (C**2 * L**2 + C * D * L**2)) 
        quotient4 = (C * L)/(C**2 * L**2 + C * D * L**2)
        quotient5 = (D * L)/(2 * (C**2 * L**2 + C * D * L**2))

        x = presign * np.sqrt( quotient1 + quotient2 + presign2*(numerator3 / denom3) + quotient4 + quotient5 )

        return x / (2 * np.pi)
    

    def total_impedance_imag(self, fc, C=None, L=None, R=None, Cc=None, L_junk=None):
        # where fc is the carrier frequency
        # Compute total device impedance at fc,
        # including the coupling capacitor in series with
        # the parallel RLC resonance

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
    
        if quantity == 'SUM':
            Zfunc = self.total_impedance_component_sum
        else:
            Zfunc = self.total_impedance_imag
    
        guess = 1./(2*np.pi*np.sqrt(self.L*(self.C + self.Cc)))
        # print(self.Lk, self.Lg, self.L, self.C, guess)
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
        # print(lbound, ubound)
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


    def compute_Qc(self, Z0=50):
        '''
        From McCarrick thesis
        
        Z0 is the characteristic impedance of the feedline
        '''
        fr = self.compute_fr()
        Qc = (8 * self.C) / (self.Cc**2 * (2 * np.pi * fr * Z0) )
        return Qc

    def fit_for_Q_values(self, span=300e3, npts=1000):
        '''
        use pete's asymmetric lorentzian fitter to estimate Q values for this resonator
        TODO this should be replaced with a generic fit wrapper
        '''

        import hidfmux.analysis.fit_resonances as fit_resonances

        fr = self.compute_fr()
        frange = np.linspace(fr-span, fr+span, npts)
        Vout = self.compute_Vout(frange)
        fit_dict = fit_resonances.fit_skewed(frange, Vout)

        Qr = fit_dict['Qr']
        Qi = fit_dict['Qi']
        Qc = fit_dict['Qc']
        return Qr, Qi, Qc
        
        
    

    #####
    # extras
    #####
    
    def generate_res_param_string(self):
        res_param_string = 'Lk=%.2e H, Lg=%.2e H, C=%.2e F, Cc=%.2e F, R=%.2e ohm'%(self.Lk, self.Lg, self.C, self.Cc, self.R)
        return res_param_string
    
    
    def plot_resonator_response(self, span=500e3, npts=1000):
        # plot a resonance
        # move the carrier frequency some distance away from the resonant frequency
        # recompute a new dV/dL, dwr/dL for some infinitesimal dL

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

        ax.set_title('Resonator frequency response\n$f_r$ = %d MHz; $V_{in}$=%.1f $\mu V$'%(1e-6*fr, 1e6*self.Vin.real))

        ax.set_ylabel('V$_{out}$')
        ax.set_xlabel('Freq. offset from f$_r$ [kHz]')

        ax.legend(loc='lower right')

        fig.tight_layout()





