'''
find the dynamic nonequilibrium impedances of a resonator, using a 
system sampling rate and a "nature" sampling rate

I^2 prioritized


maclean.rouble@mail.mcgill.ca
'''


import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import deque


from mr_complex_resonator import MR_complex_resonator as MR_complex_resonator
from mr_lekid import MR_LEKID as MR_LEKID
import utils


def make_nature_fs_timestreams(carrier_Vin_timestream, carrier_freq_timestream, system_fs, nature_fs):
    sampling_rate_factor = nature_fs // system_fs
    nature_Vin_timestream = np.repeat(carrier_Vin_timestream, sampling_rate_factor)
    nature_freq_timestream = np.repeat(carrier_freq_timestream, sampling_rate_factor)
    return nature_Vin_timestream, nature_freq_timestream

def single_resonance_iterator(res, carrier_freq_timestream,
                          carrier_Vin_timestream, 
                           nqp_timestream=None,
                          watcher_span = 100e3, nwatchers=1000, 
                          Icrit=1e-3,
                          nonlinear_R=False, alpha_R=1e3,
                          save_watcher_data=True,
                          nature_fs=1e7, system_fs=1e4,
                          stabilize_first=True,
                          stop_when_comb_hithered=False, comb_hither_threshold=1e3,
                          gather_calsamps=True):
    '''
    NOTE the assumption is the timestreams will be sampled at the nature sampling rate
    '''
    fix_Lg = res.lekid.Lg
    fr = res.lekid.compute_fr()

    Qr, Qi, Qc = res.lekid.fit_for_Q_values()
    nominal_tau = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
#     print('Qr: %.2e, nominal tau %.2e, 1/tau %.2e'%(Qr, nominal_tau, 1./nominal_tau))
    
    watcher_frange = np.linspace(fr-watcher_span, fr+watcher_span, nwatchers)

    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = res.lekid.get_att_vals(res.lekid.input_atten_dB)
    ZLNA = res.lekid.ZLNA

    # prepare the various input timestreams
    
    # from the inputs in system time, create timestreams in nature time
    timestream_len_s = len(carrier_Vin_timestream) / system_fs
    ### TODO note that if this is not inherently an integer the sampling rate interpolatoin won't be quite correct
    sample_rate_factor = nature_fs // system_fs 
    if nqp_timestream is None: 
        nqp_timestream = np.ones(len(carrier_Vin_timestream))*res.calc_nqp()

    # get the initial impedances and current etc at the probe position
    Zres = res.lekid.total_impedance(carrier_freq_timestream[0])
    ZRLC = res.lekid.parallel_RLC(carrier_freq_timestream[0])
    Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres)
    Iin = carrier_Vin_timestream[0] / (Zpar + r2)
    Ires = Iin * Zpar/Zres
    ZL = 2.j*np.pi*(res.lekid.Lk + res.lekid.Lg)*carrier_freq_timestream[0]
    IL = Ires * ZRLC / (ZL + res.lekid.R)

    outdict = {}
    outdict['system_fs'] = {}
    save_nature_data = True
    if save_nature_data: # eventually we probably want to throw this away to conserve space
        outdict['nature_fs'] = {}
    stopflag = False
    n_stabilize_iters = 500
    stabilize_iter = 0
    step = 0
    system_step = 0


    

    while step < len(nqp_timestream):
        ttt = 1./nature_fs
        if stabilize_first and stabilize_iter < n_stabilize_iters:
            step = 0 # just keep overwriting the initial entry
            system_step = 0
            stabilize_iter += 1
#         if stabilize_iter >= n_stabilize_iters:
#             print(step, system_step)
        if stop_when_comb_hithered and stopflag:
            step -= 1
            print('comb hither!')
            break

        Vin = carrier_Vin_timestream[step]
        carrier_freq = carrier_freq_timestream[step]
        nqp = nqp_timestream[step]
        outdict['nature_fs'][step] = {}
        
        if step == 0:
            outdict['nature_fs'][step]['resonator'] = copy.deepcopy(res)
            
            outdict['system_fs'][system_step] = {}
            for param in ['Iin', 'Ires', 'Zres', 'ZRLC', 'IL', 'nqp', 'carrier_Vout', 'fr', 'fr0', 'Lk', 'R', 'Lk_Isq', 'R_Isq']:
                outdict['system_fs'][system_step][param] = 0
            if save_watcher_data:
                outdict['system_fs'][system_step]['watcher_freq'] = watcher_frange
                
        outdict['nature_fs'][step]['Iin'] = Iin
        outdict['nature_fs'][step]['Ires'] = Ires
        outdict['nature_fs'][step]['Zres'] = Zres
        outdict['nature_fs'][step]['ZRLC'] = ZRLC
        outdict['nature_fs'][step]['IL'] = IL
        
        outdict['system_fs'][system_step]['Iin'] += Iin
        outdict['system_fs'][system_step]['Ires'] += Ires
        outdict['system_fs'][system_step]['Zres'] += Zres
        outdict['system_fs'][system_step]['ZRLC'] += ZRLC
        outdict['system_fs'][system_step]['IL'] += IL 
        
        outdict['nature_fs'][step]['nqp'] = nqp
        outdict['nature_fs'][step]['carrier_Vin'] = Vin
        outdict['nature_fs'][step]['carrier_freq'] = carrier_freq
        
        outdict['system_fs'][system_step]['nqp'] += nqp
        outdict['system_fs'][system_step]['carrier_Vin'] = Vin # these actually are constant for system step so no need to avg
        outdict['system_fs'][system_step]['carrier_freq'] = carrier_freq 
        
        sigma1 = res.calc_sigma1(f=carrier_freq, nqp=nqp)
        sigma2 = res.calc_sigma2(f=carrier_freq, nqp=nqp)
        sigma = sigma1 - 1.j*sigma2

        # compute the base impedance at this frequency
        Zs = res.calc_Zs(f=carrier_freq, sigma=sigma) 
        R, Lk = res.calc_R_L(f=carrier_freq, Zs=Zs)
        outdict['nature_fs'][step]['R'] = R + res.R_spoiler
        outdict['nature_fs'][step]['Lk'] = Lk
        outdict['system_fs'][system_step]['R'] += R + res.R_spoiler
        outdict['system_fs'][system_step]['Lk'] += Lk

        # generate new lekid with these params to get the fr0 from nqp only
        lekid_params = dict(R=R+res.R_spoiler, Lk=Lk, Lg=res.Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        fr0 = new_lekid.compute_fr() # the 'base' fr, without nonlinearity
        outdict['nature_fs'][step]['fr0'] = fr0
        outdict['system_fs'][system_step]['fr0'] += fr0

        # add the nonlinear current response
        if nonlinear_R:
            R = R * (1 + alpha_R*abs(IL)**2 / Icrit**2)
        R = R + res.R_spoiler
        outdict['nature_fs'][step]['R_Isq'] = R
        outdict['system_fs'][system_step]['R_Isq'] += R
                
        Lk = Lk * (1. + abs(IL)**2 / Icrit**2)
        outdict['nature_fs'][step]['Lk_Isq'] = Lk
        outdict['system_fs'][system_step]['Lk_Isq'] += Lk

        # generate new lekid with these params including the nonlinear Lk etc
        lekid_params = dict(R=R, Lk=Lk, Lg=fix_Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        Vout = new_lekid.compute_Vout(carrier_freq)  
        outdict['nature_fs'][step]['fr'] = new_lekid.compute_fr()
        outdict['nature_fs'][step]['carrier_Vout'] = Vout        
        outdict['system_fs'][system_step]['fr'] += new_lekid.compute_fr()
        outdict['system_fs'][system_step]['carrier_Vout'] += Vout        
            
        Zres = new_lekid.total_impedance(carrier_freq)
        ZRLC = new_lekid.parallel_RLC(carrier_freq)
        ZL = 2.j*np.pi*(Lk + res.Lg)*carrier_freq
        Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
        Iin = Vin / (Zpar + r2)
        
        # update the current: compute the value it WOULD step to if it could change infinitely fast
        # then use an exponential envelope to intsead step just as far as it can get in one sampling time
        next_Ires = Iin * Zpar / Zres
        Ires = next_Ires + (Ires - next_Ires) * np.exp(-ttt / nominal_tau)
        
        ### and I guess we assume that the INDUCTOR current follows this instantly...? TODO
        next_IL = Ires * ZRLC / (ZL + R)
        IL = next_IL
        outdict['nature_fs'][step]['t'] = step * ttt

        if step % sample_rate_factor == 0:
            outdict['system_fs'][system_step]['t'] = system_step * (1./system_fs)
            
            if save_watcher_data:
                watcherval = new_lekid.compute_Vout(watcher_frange)
                outdict['system_fs'][system_step]['watcher_Vout'] = watcherval
                
            for param in ['Iin', 'Ires', 'Zres', 'ZRLC', 'IL', 'nqp', 'carrier_Vout', 'fr', 'fr0', 'Lk', 'R', 'Lk_Isq', 'R_Isq']:
                outdict['system_fs'][system_step][param] /= sample_rate_factor # save the mean value from this sampling step   
            
            if abs(outdict['system_fs'][system_step]['fr'] - carrier_freq) < comb_hither_threshold and step > 0 and stop_when_comb_hithered:
                print('comb hither! %d'%system_step)
                stopflag = True
#             if not stopflag:
    
            system_step += 1
            outdict['system_fs'][system_step] = {}
            for param in ['Iin', 'Ires', 'Zres', 'ZRLC', 'IL', 'nqp', 'carrier_Vout', 'fr', 'fr0', 'Lk', 'R', 'Lk_Isq', 'R_Isq']:
                outdict['system_fs'][system_step][param] = 0
            
        step += 1

#     print('loop done, system step: %d'%system_step)
#     print(outdict['system_fs'][system_step].keys())
    # check whether the last system timestep was completed:
    if (step-1)%sample_rate_factor: # if this is nonzero, then the step did not complete
#         print('last system step (%d) did not complete'%(step-1))
#         print(step, sample_rate_factor, (step-1)%sample_rate_factor)
        outdict['system_fs'].pop(system_step)
    
    step -= 1
    system_step -= 1
    outdict['system_fs'].pop(0)
        
#     outdict['system_fs'].pop(system_step)   
    # return the final version of the resonator, with the drive applied
    res.lekid = new_lekid
    outdict['system_fs'][system_step]['resonator'] = res
    outdict['system_fs'][system_step]['comb_hithered'] = stopflag
    
    
    return outdict




def single_resonance_iterator_with_fb(res, carrier_freq_timestream,
                           nqp_timestream=None,
                           starting_Vin=1e-5, setpoint=None, theta_rot=0, 
                           delay_nsamps=10, Icoeff=0,
                          watcher_span = 100e3, nwatchers=1000, 
                          Icrit=1e-3,
                          nonlinear_R=False, alpha_R=1e3,
                          save_watcher_data=True,
                          nature_fs=1e7, system_fs=1e4,
                          stabilize_first=True):
    '''
    NOTE the assumption is the timestreams will be sampled at the nature sampling rate
    '''
    fix_Lg = res.lekid.Lg
    fr = res.lekid.compute_fr()
    # print(yes)

    Qr, Qi, Qc = res.lekid.fit_for_Q_values()
    nominal_tau = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
#     print('Qr: %.2e, nominal tau %.2e, 1/tau %.2e'%(Qr, nominal_tau, 1./nominal_tau))
    
    watcher_frange = np.linspace(fr-watcher_span, fr+watcher_span, nwatchers)

    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = res.lekid.get_att_vals(res.lekid.input_atten_dB)
    ZLNA = res.lekid.ZLNA

    # prepare the various input timestreams
    
    # from the inputs in system time, create timestreams in nature time
    timestream_len_s = len(carrier_freq_timestream) / system_fs
    ### TODO note that if this is not inherently an integer the sampling rate interpolatoin won't be quite correct
    sample_rate_factor = nature_fs // system_fs 
    # if nqp_timestream is None: 
    #     nqp_timestream = np.ones(len(carrier_freq_timestream))*res.calc_nqp()

    # get the initial impedances and current etc at the probe position
    Vin = starting_Vin
    Zres = res.lekid.total_impedance(carrier_freq_timestream[0])
    ZRLC = res.lekid.parallel_RLC(carrier_freq_timestream[0])
    Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres)
    Iin = Vin / (Zpar + r2)
    Ires = Iin * Zpar/Zres
    ZL = 2.j*np.pi*(res.lekid.Lk + res.lekid.Lg)*carrier_freq_timestream[0]
    IL = Ires * ZRLC / (ZL + res.lekid.R)

    outdict = {}
    outdict['system_fs'] = {}
    save_nature_data = True
    if save_nature_data: # eventually we probably want to throw this away to conserve space
        outdict['nature_fs'] = {}
    stopflag = False
    n_stabilize_iters = 500
    stabilize_iter = 0
    step = 0
    system_step = 0

    # set up the feedback parameters
    meas_buf = deque([setpoint]*delay_nsamps, maxlen=delay_nsamps) # let's just use a single delay for now
    acc = 0

    while step < len(nqp_timestream):
        ttt = 1./nature_fs
        if stabilize_first and stabilize_iter < n_stabilize_iters:
            step = 0 # just keep overwriting the initial entry
            system_step = 0
            stabilize_iter += 1


        # Vin = carrier_Vin_timestream[step]
        carrier_freq = carrier_freq_timestream[step]
        nqp = nqp_timestream[step]
        outdict['nature_fs'][step] = {}
        
        if step == 0:
            outdict['nature_fs'][step]['resonator'] = copy.deepcopy(res)
            
            outdict['system_fs'][system_step] = {}
            for param in ['Iin', 'Ires', 'Zres', 'ZRLC', 'IL', 'nqp', 'carrier_Vout', 'fr', 'fr0', 'Lk', 'R', 'Lk_Isq', 'R_Isq']:
                outdict['system_fs'][system_step][param] = 0
            if save_watcher_data:
                outdict['system_fs'][system_step]['watcher_freq'] = watcher_frange
                
        outdict['nature_fs'][step]['Iin'] = Iin
        outdict['nature_fs'][step]['Ires'] = Ires
        outdict['nature_fs'][step]['Zres'] = Zres
        outdict['nature_fs'][step]['ZRLC'] = ZRLC
        outdict['nature_fs'][step]['IL'] = IL
        
        outdict['system_fs'][system_step]['Iin'] += Iin
        outdict['system_fs'][system_step]['Ires'] += Ires
        outdict['system_fs'][system_step]['Zres'] += Zres
        outdict['system_fs'][system_step]['ZRLC'] += ZRLC
        outdict['system_fs'][system_step]['IL'] += IL 
        
        outdict['nature_fs'][step]['nqp'] = nqp
        outdict['nature_fs'][step]['carrier_Vin'] = Vin
        outdict['nature_fs'][step]['carrier_freq'] = carrier_freq
        
        outdict['system_fs'][system_step]['nqp'] += nqp
        outdict['system_fs'][system_step]['carrier_Vin'] = Vin # these actually are constant for system step so no need to avg
        outdict['system_fs'][system_step]['carrier_freq'] = carrier_freq 
        
        sigma1 = res.calc_sigma1(f=carrier_freq, nqp=nqp)
        sigma2 = res.calc_sigma2(f=carrier_freq, nqp=nqp)
        sigma = sigma1 - 1.j*sigma2

        # compute the base impedance at this frequency
        Zs = res.calc_Zs(f=carrier_freq, sigma=sigma) 
        R, Lk = res.calc_R_L(f=carrier_freq, Zs=Zs)
        outdict['nature_fs'][step]['R'] = R + res.R_spoiler
        outdict['nature_fs'][step]['Lk'] = Lk
        outdict['system_fs'][system_step]['R'] += R + res.R_spoiler
        outdict['system_fs'][system_step]['Lk'] += Lk

        # generate new lekid with these params to get the fr0 from nqp only
        lekid_params = dict(R=R+res.R_spoiler, Lk=Lk, Lg=res.Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        fr0 = new_lekid.compute_fr() # the 'base' fr, without nonlinearity
        outdict['nature_fs'][step]['fr0'] = fr0
        outdict['system_fs'][system_step]['fr0'] += fr0

        # add the nonlinear current response
        if nonlinear_R:
            R = R * (1 + alpha_R*abs(IL)**2 / Icrit**2)
        R = R + res.R_spoiler
        outdict['nature_fs'][step]['R_Isq'] = R
        outdict['system_fs'][system_step]['R_Isq'] += R
                
        Lk = Lk * (1. + abs(IL)**2 / Icrit**2)
        outdict['nature_fs'][step]['Lk_Isq'] = Lk
        outdict['system_fs'][system_step]['Lk_Isq'] += Lk

        # generate new lekid with these params including the nonlinear Lk etc
        lekid_params = dict(R=R, Lk=Lk, Lg=fix_Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        Vout = new_lekid.compute_Vout(carrier_freq) * np.exp(1.j*theta_rot)
        outdict['nature_fs'][step]['fr'] = new_lekid.compute_fr()
        outdict['nature_fs'][step]['carrier_Vout'] = Vout        
        outdict['system_fs'][system_step]['fr'] += new_lekid.compute_fr()
        outdict['system_fs'][system_step]['carrier_Vout'] += Vout        
            
        Zres = new_lekid.total_impedance(carrier_freq)
        ZRLC = new_lekid.parallel_RLC(carrier_freq)
        ZL = 2.j*np.pi*(Lk + res.Lg)*carrier_freq
        Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
        Iin = Vin / (Zpar + r2)
        
        # update the current: compute the value it WOULD step to if it could change infinitely fast
        # then use an exponential envelope to intsead step just as far as it can get in one sampling time
        next_Ires = Iin * Zpar / Zres
        Ires = next_Ires + (Ires - next_Ires) * np.exp(-ttt / nominal_tau)
        
        ### and I guess we assume that the INDUCTOR current follows this instantly...? TODO
        next_IL = Ires * ZRLC / (ZL + R)
        IL = next_IL
        outdict['nature_fs'][step]['t'] = step * ttt

        if step % sample_rate_factor == 1:
            outdict['system_fs'][system_step]['t'] = system_step * (1./system_fs)
            
            if save_watcher_data:
                watcherval = new_lekid.compute_Vout(watcher_frange)
                outdict['system_fs'][system_step]['watcher_Vout'] = watcherval
                
            for param in ['Iin', 'Ires', 'Zres', 'ZRLC', 'IL', 'nqp', 'carrier_Vout', 'fr', 'fr0', 'Lk', 'R', 'Lk_Isq', 'R_Isq']:
                outdict['system_fs'][system_step][param] /= sample_rate_factor # save the mean value from this sampling step   

            ## update the feedback
            Vout = outdict['system_fs'][system_step]['carrier_Vout']
            meas_buf.append(Vout)
            Vout_delayed = meas_buf[0]
            ttt_sys = 1./system_fs
            acc += (setpoint.imag - Vout_delayed.imag*Vin) * (ttt_sys) * Icoeff 
            Vin = starting_Vin * (1 + acc)
            outdict['system_fs'][system_step]['acc'] = acc
    
            system_step += 1
            outdict['system_fs'][system_step] = {}
            for param in ['Iin', 'Ires', 'Zres', 'ZRLC', 'IL', 'nqp', 'carrier_Vout', 'fr', 'fr0', 'Lk', 'R', 'Lk_Isq', 'R_Isq']:
                outdict['system_fs'][system_step][param] = 0
            
        step += 1


    # check whether the last system timestep was completed:
    if (step-1)%sample_rate_factor: # if this is nonzero, then the step did not complete
#         print('last system step (%d) did not complete'%(step-1))
#         print(step, sample_rate_factor, (step-1)%sample_rate_factor)
        outdict['system_fs'].pop(system_step)
    
    step -= 1
    system_step -= 1
    outdict['system_fs'].pop(0)
        
#     outdict['system_fs'].pop(system_step)   
    # return the final version of the resonator, with the drive applied
    res.lekid = new_lekid
    outdict['system_fs'][system_step]['resonator'] = res
    # outdict['system_fs'][system_step]['comb_hithered'] = stopflag
    
    
    return outdict

