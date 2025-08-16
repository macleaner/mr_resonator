'''
find the dynamic nonequilibrium impedances of a resonator

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


def single_resonance_iterator(res, carrier_freq_timestream,
                          carrier_Vin_timestream, nqp_timestream=None,
                          watcher_span = 100e3, nwatchers=1000, 
                          Icrit=1e-3,
                          nonlinear_R=False, alpha_R=1e3,
                          save_watcher_data=True,
                          Ires_fs=1e7, stabilize_first=True,
                          stop_when_comb_hithered=False, comb_hither_threshold=1e3,
                          gather_calsamps=True):
    '''
    where we simplify things by supplying an nqp timestream and pretending that it translates instantly into
    a value for Lk
    ideally this timestream is probably a step function or something similarly simple
    note that the initial value in the nqp timestream should be the baseline nqp value for the given resonance, with 
    its temperature and loading etc

    otherwise to do a more realistic nqp timestream, we would need to consider that the nqp itself doesn't change
    instantly, but instead is governed by some time constant. Although once it does change we would still assume that
    Lk changes instantly in response.

    So anyway basically we are just investigating the time constant associated with the change in Ires, which reuslts
    from this change in Lk.
    '''
    fix_Lg = res.lekid.Lg
    Vin = res.lekid.Vin
    lekid = res.lekid
    fr = lekid.compute_fr()


    Qr, Qi, Qc = res.lekid.fit_for_Q_values()
    nominal_tau = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
#     print('Qr: %.2e, nominal tau %.2e, 1/tau %.2e'%(Qr, nominal_tau, 1./nominal_tau))
    
    watcher_frange = np.linspace(fr-watcher_span, fr+watcher_span, nwatchers)

    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = res.lekid.get_att_vals(res.lekid.input_atten_dB)
    ZLNA = res.lekid.ZLNA

    if nqp_timestream is None:
        nqp_timestream = np.ones(len(carrier_Vin_timestream))*res.calc_nqp()

    # get the initial impedances and current etc at the probe position
    Zres = res.lekid.total_impedance(carrier_freq_timestream[0])
    ZRLC = res.lekid.parallel_RLC(carrier_freq_timestream[0])
    Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres)
    Iin = Vin / (Zpar + r2)
    Ires = Iin * Zpar/Zres
    ZL = 2.j*np.pi*(res.lekid.Lk + res.lekid.Lg)*carrier_freq_timestream[0]
    IL = Ires * ZRLC / (ZL + res.lekid.R)

    outdict = {}
    stopflag = False
    n_stabilize_iters = 500
    stabilize_iter = 0
    step = 0

    while step < len(nqp_timestream):
        if stabilize_first and stabilize_iter < n_stabilize_iters:
            step = 0 # just keep overwriting the initial entry
            stabilize_iter += 1

        if stop_when_comb_hithered and stopflag:
            step -= 1
            print('comb hither!')
            break

        Vin = carrier_Vin_timestream[step]
        carrier_freq = carrier_freq_timestream[step]
        nqp = nqp_timestream[step]
        outdict[step] = {}
        if step == 0:
            outdict[step]['resonator'] = copy.deepcopy(res)
            if save_watcher_data:
                outdict[step]['watcher_freq'] = watcher_frange
                
        
        outdict[step]['Iin'] = Iin
        outdict[step]['Ires'] = Ires
        outdict[step]['Zres'] = Zres
        outdict[step]['ZRLC'] = ZRLC
        outdict[step]['IL'] = IL
        
        outdict[step]['nqp'] = nqp
        outdict[step]['carrier_Vin'] = Vin
        outdict[step]['carrier_freq'] = carrier_freq # duplicated
        
        sigma1 = res.calc_sigma1(f=carrier_freq, nqp=nqp)
        sigma2 = res.calc_sigma2(f=carrier_freq, nqp=nqp)
        sigma = sigma1 - 1.j*sigma2

        # compute the base impedance at this frequency
        Zs = res.calc_Zs(f=carrier_freq, sigma=sigma) 
        R, Lk = res.calc_R_L(f=carrier_freq, Zs=Zs)
        outdict[step]['R'] = R + res.R_spoiler
        outdict[step]['Lk'] = Lk

        # generate new lekid with these params to get the fr0 from nqp only
        lekid_params = dict(R=R+res.R_spoiler, Lk=Lk, Lg=res.Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        fr0 = new_lekid.compute_fr() # the 'base' fr, without nonlinearity
        outdict[step]['fr0'] = fr0

        # add the nonlinear current response
        if nonlinear_R:
            R = R * (1 + alpha_R*abs(IL)**2 / Icrit**2)
        R = R + res.R_spoiler
        outdict[step]['R_Isq'] = R
                
        Lk = Lk * (1. + abs(IL)**2 / Icrit**2)
        outdict[step]['Lk_Isq'] = Lk

        # generate new lekid with these params including the nonlinear Lk etc
        lekid_params = dict(R=R, Lk=Lk, Lg=fix_Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        Vout = new_lekid.compute_Vout(carrier_freq) ###
        
        watcherval = new_lekid.compute_Vout(watcher_frange)
        outdict[step]['fr'] = new_lekid.compute_fr()
        outdict[step]['carrier_Vout'] = Vout
        if save_watcher_data:
            outdict[step]['watcher_Vout'] = watcherval
            
        Zres = new_lekid.total_impedance(carrier_freq)
        ZRLC = new_lekid.parallel_RLC(carrier_freq)
        ZL = 2.j*np.pi*(Lk + res.Lg)*carrier_freq
        Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
        Iin = Vin / (Zpar + r2)
        
        # update the current: compute the value it WOULD step to if it could change infinitely fast
        # then use an exponential envelope to intsead step just as far as it can get in one sampling time
        next_Ires = Iin * Zpar / Zres
        ttt = 1./Ires_fs
        Ires = next_Ires + (Ires - next_Ires) * np.exp(-ttt / nominal_tau)
        
        Vout = Ires * Zres
        outdict[step]['carrier_Vout_IresZres'] = Vout
        
        ### and I guess we assume that the INDUCTOR current follows this instantly...? TODO
        next_IL = Ires * ZRLC / (ZL + R)
        IL = next_IL
        # I_L = next_I_L + (I_L - next_I_L) * np.exp(-ttt / nominal_tau)
        outdict[step]['t'] = step * ttt

        if abs(outdict[step]['fr'] - carrier_freq) < comb_hither_threshold and step > 0:
            # print('hit threshold at astep', astep, 'stabilize_iter:', stabilize_iter)
            stopflag = True
        step += 1

    # print('finished loop. astep:', astep)
    # print(stopflag)
    if not stopflag or not stop_when_comb_hithered:
        step -= 1
    # return the final version of the resonator, with the drive applied
    res.lekid = new_lekid
    outdict[step]['resonator'] = res
    outdict[step]['comb_hithered'] = stopflag
    
    if gather_calsamps:
        nqp_timestream = np.random.normal(res.calc_nqp(), 3., 100)
        calsamps = []
        for cstep in range(len(nqp_timestream)):
            nqp = nqp_timestream[cstep]
            sigma1 = res.calc_sigma1(f=carrier_freq, nqp=nqp)
            sigma2 = res.calc_sigma2(f=carrier_freq, nqp=nqp)
            sigma = sigma1 - 1.j*sigma2

            # compute the base impedance at this frequency
            Zs = res.calc_Zs(f=carrier_freq, sigma=sigma) 
            R, Lk = res.calc_R_L(f=carrier_freq, Zs=Zs)

            # add the nonlinear current response
            if nonlinear_R:
                R = R * (1 + alpha_R*abs(IL)**2 / Icrit**2)
            R = R + res.R_spoiler
            Lk = Lk * (1. + abs(IL)**2 / Icrit**2)

            # generate new lekid with these params including the nonlinear Lk etc
            lekid_params = dict(R=R, Lk=Lk, Lg=fix_Lg, C=res.C, Cc=res.Cc, Vin=Vin)
            new_lekid = MR_LEKID(**lekid_params)
            Vout = new_lekid.compute_Vout(carrier_freq)
            calsamps.append(Vout)
            
            Zres = new_lekid.total_impedance(carrier_freq)
            ZRLC = new_lekid.parallel_RLC(carrier_freq)
            ZL = 2.j*np.pi*(Lk + res.Lg)*carrier_freq
            Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
            Iin = Vin / (Zpar + r2)

            # update the current: compute the value it WOULD step to if it could change infinitely fast
            # then use an exponential envelope to intsead step just as far as it can get in one sampling time
            next_Ires = Iin * Zpar / Zres
            ttt = 1./Ires_fs
            Ires = next_Ires + (Ires - next_Ires) * np.exp(-ttt / nominal_tau)

            ### and I guess we assume that the INDUCTOR current follows this instantly...? TODO
            next_IL = Ires * ZRLC / (ZL + R)
            IL = next_IL

    outdict[step]['calsamps'] = np.asarray(calsamps)
    return outdict



def single_resonance_iterator_with_fb(
                          res, carrier_freq_timestream,
                          nqp_timestream=None, 
                            starting_Vin=1e-5,
                          Icoeff=1., setpoint=None, theta_rot=0,
                          fb_fs=None, delay_nsamps=10,
                          watcher_span = 100e3, nwatchers=1000, 
                          Icrit=1e-3,
                          nonlinear_R=False, alpha_R=1e3,
                          save_watcher_data=True,
                          Ires_fs=1e7, stabilize_first=True):
    '''

    '''
    if fb_fs == None: ####
        fb_fs = Ires_fs
    meas_buf = deque([0.]*delay_nsamps, maxlen=delay_nsamps) # let's just use a single delay for now
        
    fix_Lg = res.lekid.Lg
    Vin = starting_Vin
    lekid = res.lekid
    fr = lekid.compute_fr()

    Qr, Qi, Qc = res.lekid.fit_for_Q_values()
    nominal_tau = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
    print('Qr: %.2e, nominal tau %.2e, 1/tau %.2e'%(Qr, nominal_tau, 1./nominal_tau))
    
    watcher_frange = np.linspace(fr-watcher_span, fr+watcher_span, nwatchers)

    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = res.lekid.get_att_vals(res.lekid.input_atten_dB)
    ZLNA = res.lekid.ZLNA

    if nqp_timestream is None:
        nqp_timestream = np.ones(len(carrier_freq_timestream))*res.calc_nqp()

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
    stopflag = False
    n_stabilize_iters = 500
    stabilize_iter = 0
    step = 0
    acc = 0
    

    while step < len(nqp_timestream):
        ttt = 1./Ires_fs
        if stabilize_first and stabilize_iter < n_stabilize_iters:
            step = 0 # just keep overwriting the initial entry
            stabilize_iter += 1

        carrier_freq = carrier_freq_timestream[step]
        nqp = nqp_timestream[step]
        outdict[step] = {}
        if step == 0:
            outdict[step]['resonator'] = copy.deepcopy(res)
            if save_watcher_data:
                outdict[step]['watcher_freq'] = watcher_frange
                
        outdict[step]['carrier_Vin'] = Vin
        outdict[step]['Iin'] = Iin
        outdict[step]['Ires'] = Ires
        outdict[step]['Zres'] = Zres
        outdict[step]['ZRLC'] = ZRLC
        outdict[step]['IL'] = IL
        
        outdict[step]['nqp'] = nqp
#         outdict[step]['carrier_Vin'] = Vin
        outdict[step]['carrier_freq'] = carrier_freq # duplicated
        
        sigma1 = res.calc_sigma1(f=carrier_freq, nqp=nqp)
        sigma2 = res.calc_sigma2(f=carrier_freq, nqp=nqp)
        sigma = sigma1 - 1.j*sigma2

        # compute the base impedance at this frequency
        Zs = res.calc_Zs(f=carrier_freq, sigma=sigma) 
        R, Lk = res.calc_R_L(f=carrier_freq, Zs=Zs)
        outdict[step]['R'] = R + res.R_spoiler
        outdict[step]['Lk'] = Lk

        # generate new lekid with these params to get the fr0 from nqp only
        lekid_params = dict(R=R+res.R_spoiler, Lk=Lk, Lg=res.Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        fr0 = new_lekid.compute_fr() # the 'base' fr, without nonlinearity
        outdict[step]['fr0'] = fr0

        # add the nonlinear current response
        if nonlinear_R:
            R = R * (1 + alpha_R*abs(IL)**2 / Icrit**2)
        R = R + res.R_spoiler
        outdict[step]['R_Isq'] = R
                
        Lk = Lk * (1. + abs(IL)**2 / Icrit**2)
        outdict[step]['Lk_Isq'] = Lk

        # generate new lekid with these params including the nonlinear Lk etc
        lekid_params = dict(R=R, Lk=Lk, Lg=fix_Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        watcherval = new_lekid.compute_Vout(watcher_frange)
        outdict[step]['fr'] = new_lekid.compute_fr()
        if save_watcher_data:
            outdict[step]['watcher_Vout'] = watcherval
            
        Vout = new_lekid.compute_Vout(carrier_freq)
#         Vout *= np.exp(1.j*theta_rot)
        outdict[step]['carrier_Vout'] = Vout
        meas_buf.append(Vout)
        Vout_delayed = meas_buf[0]
        acc += (setpoint.imag*Vin - Vout_delayed.imag) * (ttt) * Icoeff 
        Vin = starting_Vin * (1 + acc)
        outdict[step]['acc'] = acc
        
            
        Zres = new_lekid.total_impedance(carrier_freq)
        ZRLC = new_lekid.parallel_RLC(carrier_freq)
        ZL = 2.j*np.pi*(Lk + res.Lg)*carrier_freq
        Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
        Iin = Vin / (Zpar + r2)

        # update the current: compute the value it WOULD step to if it could change infinitely fast
        # then use an exponential envelope to intsead step just as far as it can get in one sampling time
        next_Ires = Iin * Zpar / Zres
        Ires = next_Ires + (Ires - next_Ires) * np.exp(-ttt / nominal_tau)
        
        Vout = Ires * Zres
        outdict[step]['carrier_Vout_IresZres'] = Vout

        ### and I guess we assume that the INDUCTOR current follows this instantly...? TODO
        next_IL = Ires * ZRLC / (ZL + R)
        IL = next_IL
        # I_L = next_I_L + (I_L - next_I_L) * np.exp(-ttt / nominal_tau)
        outdict[step]['t'] = step * ttt

        step += 1

    # print('finished loop. astep:', astep)
    # print(stopflag)
    if not stopflag:
        step -= 1
    # return the final version of the resonator, with the drive applied
    res.lekid = new_lekid
    outdict[step]['resonator'] = res
    
    return outdict
    
    
    
    
    
    
def single_resonance_iterator_with_fb_outdict(setup_outdict, 
                          nqp_timestream=None, 
                          Icoeff=1., #setpoint=None, theta_rot=0,
                          fb_fs=None, delay_nsamps=10,
                          watcher_span = 100e3, nwatchers=1000, 
                          Icrit=1e-3,
                          nonlinear_R=False, alpha_R=1e3,
                          save_watcher_data=True,
                          Ires_fs=1e7, stabilize_first=False,
                                             plot_calsamps=False):
    '''

    '''
    
    
    res = setup_outdict['resonator']
    calsamps = setup_outdict['calsamps']
    
    if plot_calsamps:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121)
        bx = fig.add_subplot(122)
        ax.plot(calsamps.real-np.mean(calsamps.real))
        ax.plot(calsamps.imag-np.mean(calsamps.imag))
        bx.scatter(calsamps.real, calsamps.imag)
    
    
    calsamps, theta_rot = utils.rotate_iq_plane(calsamps)
    Vout_setpoint = np.mean(calsamps)
    if plot_calsamps:
        ax.plot(calsamps.real-np.mean(calsamps.real), '--', color='tab:blue', alpha=0.7)
        ax.plot(calsamps.imag-np.mean(calsamps.imag), '--', color='tab:orange', alpha=0.7)
        bx.scatter(calsamps.real, calsamps.imag)
        bx.scatter(Vout_setpoint.real, Vout_setpoint.imag, marker='x', s=200, label='setpoint')
        prev_Vout = setup_outdict['carrier_Vout']
        bx.scatter(prev_Vout.real, prev_Vout.imag, marker='*', s=200)
        prev_Vout *= np.exp(1.j*theta_rot)
        bx.scatter(prev_Vout.real, prev_Vout.imag, marker='*', s=200)
        utils.square_axes(bx)
        bx.legend()
        fig.suptitle('calibration samples')
        
    
    carrier_freq = setup_outdict['carrier_freq']
    starting_Vin = setup_outdict['carrier_Vin']
    Vin = starting_Vin 
    Vout_setpoint *= starting_Vin
    meas_buf = deque([Vout_setpoint]*delay_nsamps, maxlen=delay_nsamps) # let's just use a single delay for now
    print(meas_buf)

    fix_Lg = res.lekid.Lg
    fr = res.lekid.compute_fr()
    Qr, Qi, Qc = res.lekid.fit_for_Q_values()
    nominal_tau = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
    print('Qr: %.2e, nominal tau %.2e, 1/tau %.2e'%(Qr, nominal_tau, 1./nominal_tau))
    
    watcher_frange = np.linspace(fr-watcher_span, fr+watcher_span, nwatchers)

    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = res.lekid.get_att_vals(res.lekid.input_atten_dB)
    ZLNA = res.lekid.ZLNA

    if nqp_timestream is None:
        nqp_timestream = np.ones(len(carrier_freq_timestream))*res.calc_nqp()

    # get the initial impedances and current etc at the probe position
    Iin = setup_outdict['Iin']
    Zres = setup_outdict['Zres']
    ZRLC = setup_outdict['ZRLC']
    Ires = setup_outdict['Ires']
    IL = setup_outdict['IL']

    outdict = {}
    stopflag = False
    n_stabilize_iters = 500
    stabilize_iter = 0
    step = 0
    acc = 0
    
    while step < len(nqp_timestream):
        ttt = 1./Ires_fs
        if stabilize_first and stabilize_iter < n_stabilize_iters:
            step = 0 # just keep overwriting the initial entry
            stabilize_iter += 1

        nqp = nqp_timestream[step]
        outdict[step] = {}
        if step == 0:
            outdict[step]['resonator'] = copy.deepcopy(res)
            if save_watcher_data:
                outdict[step]['watcher_freq'] = watcher_frange
                
        outdict[step]['Vout_setpoint'] = Vout_setpoint# * Vin        
        outdict[step]['carrier_Vin'] = Vin
        outdict[step]['Iin'] = Iin
        outdict[step]['Ires'] = Ires
        outdict[step]['Zres'] = Zres
        outdict[step]['ZRLC'] = ZRLC
        outdict[step]['IL'] = IL
        
        outdict[step]['nqp'] = nqp
        outdict[step]['carrier_freq'] = carrier_freq # duplicated
        
        sigma1 = res.calc_sigma1(f=carrier_freq, nqp=nqp)
        sigma2 = res.calc_sigma2(f=carrier_freq, nqp=nqp)
        sigma = sigma1 - 1.j*sigma2

        # compute the base impedance at this frequency
        Zs = res.calc_Zs(f=carrier_freq, sigma=sigma) 
        R, Lk = res.calc_R_L(f=carrier_freq, Zs=Zs)
        outdict[step]['R'] = R + res.R_spoiler
        outdict[step]['Lk'] = Lk

        # generate new lekid with these params to get the fr0 from nqp only
        lekid_params = dict(R=R+res.R_spoiler, Lk=Lk, Lg=res.Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        fr0 = new_lekid.compute_fr() # the 'base' fr, without nonlinearity
        outdict[step]['fr0'] = fr0

        # add the nonlinear current response
        if nonlinear_R:
            R = R * (1 + alpha_R*abs(IL)**2 / Icrit**2)
        R = R + res.R_spoiler
        outdict[step]['R_Isq'] = R
                
        Lk = Lk * (1. + abs(IL)**2 / Icrit**2)
        outdict[step]['Lk_Isq'] = Lk

        # generate new lekid with these params including the nonlinear Lk etc
        lekid_params = dict(R=R, Lk=Lk, Lg=fix_Lg, C=res.C, Cc=res.Cc, Vin=Vin)
        new_lekid = MR_LEKID(**lekid_params)
        watcherval = new_lekid.compute_Vout(watcher_frange)
        watcherval *= np.exp(1.j*theta_rot)
        outdict[step]['fr'] = new_lekid.compute_fr()
        if save_watcher_data:
            outdict[step]['watcher_Vout'] = watcherval
            
        Vout = new_lekid.compute_Vout(carrier_freq)
        Vout *= np.exp(1.j*theta_rot)
        outdict[step]['carrier_Vout'] = Vout
        meas_buf.append(Vout)
        Vout_delayed = meas_buf[0]
        ### TODO should we used a delayed Vin, or the ACTUAL Vin now?
        acc += (Vout_setpoint.imag - Vout_delayed.imag*Vin) * (ttt) * Icoeff 
        Vin = starting_Vin * (1 + acc)
        outdict[step]['acc'] = acc
        
            
        Zres = new_lekid.total_impedance(carrier_freq)
        ZRLC = new_lekid.parallel_RLC(carrier_freq)
        ZL = 2.j*np.pi*(Lk + res.Lg)*carrier_freq
        Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
        Iin = Vin / (Zpar + r2)

        # update the current: compute the value it WOULD step to if it could change infinitely fast
        # then use an exponential envelope to intsead step just as far as it can get in one sampling time
        next_Ires = Iin * Zpar / Zres
        Ires = next_Ires + (Ires - next_Ires) * np.exp(-ttt / nominal_tau)
        Vout = Ires * Zres
        outdict[step]['carrier_Vout_IresZres'] = Vout * np.exp(1.j*theta_rot)

        ### and I guess we assume that the INDUCTOR current follows this instantly...? TODO
        next_IL = Ires * ZRLC / (ZL + R)
        IL = next_IL
        outdict[step]['t'] = step * ttt

        step += 1

    if not stopflag:
        step -= 1
    # return the final version of the resonator, with the drive applied
    res.lekid = new_lekid
    outdict[step]['resonator'] = res
    
    return outdict