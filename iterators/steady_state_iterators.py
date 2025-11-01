'''
iteratively solve the impedances of a resonance in steady state

I^2 prioritized


maclean.rouble@mail.mcgill.ca
'''


import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate as interpolate

from mr_complex_resonator import MR_complex_resonator as MR_complex_resonator
from mr_lekid import MR_LEKID as MR_LEKID
import utils





def single_resonance_iterator(res, carrier_freq_timestream, carrier_Vin_timestream,
                    nqp_timestream=None,
                    save_watcher_data=True, watcher_span=300e3, Icrit=1e-3,
                    niters=150, damp=0.1,
                    stop_when_comb_hithered=False, comb_hither_fdetune=0, comb_hither_threshold=500,
                    verbose=False):
    '''
    comb_hither_fdetune: the target frequency separation between the carrier
        and the driven resonant frequency
    '''

    target = copy.deepcopy(res)

    # probe sweep parameters
    fr = target.lekid.compute_fr()

    if nqp_timestream is None:
        nqp_timestream = np.ones(len(carrier_freq_timestream)) * target.calc_nqp()
    
    # Qr, Qi, Qc = target.lekid.fit_for_Q_values()
    # nominal_tau_targ = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
    
    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = target.lekid.get_att_vals(target.lekid.input_atten_dB)
    ZLNA = 50.
    Zatt = r3
    
    if save_watcher_data:
        watcher_freq = np.linspace(fr-watcher_span, fr+watcher_span, 1000)
    
    # get the initial impedances and current etc at the first probe position
    Ztarg = target.lekid.total_impedance(carrier_freq_timestream[0])
    Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg)
    Iin = carrier_Vin_timestream[0] / (Zpar + r2)
    Itarg = Iin * Zpar/Ztarg

    targ_ZL = 2.j*np.pi*(target.lekid.Lk + target.Lg)*carrier_freq_timestream[0]
    targ_ZRLC = target.lekid.parallel_RLC(carrier_freq_timestream[0])
    targ_IL = Itarg * targ_ZRLC / (targ_ZL + target.lekid.R)


    outdict = {}
    params = ['Lk', 'Lk_Isq', 'R', 'fr', 'IL', 'Ires', 'Zres']
    carrier_params = ['carrier_freq', 'carrier_Vin', 'carrier_Vout', 'Iin']
    
    stop_flag = False
    
    # start sweepin'
    for step in range(len(carrier_freq_timestream)):

        outdict[step] = {}
        if step == 0:
            outdict[0]['resonator'] = target
            if save_watcher_data:
                outdict[0]['watcher_freq'] = watcher_freq # just save this once since it's always the same
                
        carrier_freq = carrier_freq_timestream[step]
        carrier_Vin = carrier_Vin_timestream[step]
        outdict[step]['carrier_freq'] = carrier_freq
        outdict[step]['carrier_Vin'] = carrier_Vin
        outdict[step]['carrier_Vout'] = {}
        outdict[step]['carrier_Vout']['iters'] = []
        outdict[step]['Iin'] = {}
        outdict[step]['Iin']['iters'] = []

        nqp = nqp_timestream[step]
        
    
        for resonator in ['targ']:
            outdict[step][resonator] = {}
            for param in params:
                outdict[step][resonator][param] = {}
                outdict[step][resonator][param]['iters'] = []

        outdict[step]['targ']['nqp'] = nqp
        

        
        # up to this point these should always be the same on each iter, since we are not changing nqp.
        for iiter in range(niters): # iterate at this carrier freq and Vin to steady state\
            outdict[step]['Iin']['iters'].append(Iin)
            outdict[step]['targ']['Zres']['iters'].append(Ztarg)
            outdict[step]['targ']['Ires']['iters'].append(Itarg)
            outdict[step]['targ']['IL']['iters'].append(targ_IL)
            
            # get the base impedances due to the qp population        
            targ_nqp = nqp
            targ_sigma1 = target.calc_sigma1(f=carrier_freq, nqp=targ_nqp)
            targ_sigma2 = target.calc_sigma2(f=carrier_freq, nqp=targ_nqp)
            targ_sigma = targ_sigma1 - 1.j*targ_sigma2

            targ_Zs = target.calc_Zs(f=carrier_freq, sigma=targ_sigma) 
            targ_R, targ_Lk = target.calc_R_L(f=carrier_freq, Zs=targ_Zs)
            targ_R = targ_R + target.R_spoiler
            outdict[step]['targ']['Lk']['iters'].append(targ_Lk)
            outdict[step]['targ']['R']['iters'].append(targ_R)
            
            # add the nonlinear current response
            targ_Lk = targ_Lk * (1. + abs(targ_IL)**2 / Icrit**2)  
            outdict[step]['targ']['Lk_Isq']['iters'].append(targ_Lk)
            
            # generate new lekid with these params
            targ_lekid_params = dict(R=targ_R, Lk=targ_Lk, Lg=target.Lg, C=target.C, Cc=target.Cc, Vin=carrier_Vin)
            targ_new_lekid = MR_LEKID(**targ_lekid_params)
            targ_fr = targ_new_lekid.compute_fr()
            outdict[step]['targ']['fr']['iters'].append(targ_fr)
            
            # compute impedances and current for next iter
            Ztarg = targ_new_lekid.total_impedance(fc=carrier_freq)
            Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg)
            Iin = carrier_Vin / (Zpar + r2)
            
            next_Itarg = Iin * Zpar/Ztarg
            Itarg = Itarg + damp*(next_Itarg - Itarg)
            
            targ_ZL = 2.j*np.pi*(targ_Lk + target.Lg)*carrier_freq
            targ_ZRLC = targ_new_lekid.parallel_RLC(carrier_freq)
            targ_IL = Itarg * targ_ZRLC / (targ_ZL + targ_R)
            
            Vout = carrier_Vin * (Zpar / (r2 + Zpar))
            outdict[step]['carrier_Vout']['iters'].append(Vout)

            
        # get the whole transfer function from the "watcher" measurement
        # only do this after the iterations or else the output is tooooo big
        if save_watcher_data:
            Ztarg = targ_new_lekid.total_impedance(fc=watcher_freq)
            Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg)
            watcher_Vout = 1. * (Zpar / ( r2 + Zpar))
            watcher_Vout *= carrier_Vin
            outdict[step]['watcher_Vout'] = watcher_Vout
            
        for resonator in ['targ']:    
            for param in params:
                outdict[step][resonator][param]['final'] = outdict[step][resonator][param]['iters'][-1]
        outdict[step]['carrier_Vout']['final'] = outdict[step]['carrier_Vout']['iters'][-1]
        
        if stop_when_comb_hithered:
            # fdetune = fc - fr --> fr = fc - fdetune
            # print('%.3e'%(abs((carrier_freq - comb_hither_fdetune) - targ_fr)))
            if abs((carrier_freq - comb_hither_fdetune) - targ_fr) < comb_hither_threshold:
            # if abs((carrier_freq - targ_fr) - comb_hither_fdetune) < comb_hither_threshold:
                stop_flag = True
                # print('Comb hither! stopping at step %d'%step)

        if stop_flag:
            break

    target.lekid = targ_new_lekid
    # TODO update other params?
    outdict[step]['resonator'] = target
    outdict[step]['Iin']['final'] = outdict[step]['Iin']['iters'][-1]
    outdict[step]['carrier_Vout']['final'] = outdict[step]['carrier_Vout']['iters'][-1]
    outdict[step]['comb_hithered'] = stop_flag
        
    return outdict








def two_resonance_iterator(target, neighbour,
                    carrier_freq_timestream, carrier_Vin_timestream,
                    targ_nqp_timestream=None, neigh_nqp_timestream=None,
                    save_watcher_data=True, watcher_span=300e3, Icrit=1e-3,
                    niters=150, damp=0.1, 
                    stop_when_comb_hithered=False, comb_hither_fdetune=0, comb_hither_threshold=500,
                    verbose=False):
    '''
    
    '''


    # probe sweep parameters
    fr = target.lekid.compute_fr()
    fneigh = neighbour.lekid.compute_fr()
    
    Qr, Qi, Qc = target.lekid.fit_for_Q_values()
    nominal_tau_targ = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
    Qr, Qi, Qc = neighbour.lekid.fit_for_Q_values()
    nominal_tau_neigh = Qr / (2*np.pi*fr) ### we are assuming this doesn't change with readout current, which is clearly wrong
    
    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = target.lekid.get_att_vals(target.lekid.input_atten_dB)
    ZLNA = 50.
    Zatt = r3
    
    if save_watcher_data:
        llim = min([fr, fneigh]) - watcher_span
        ulim = max([fr, fneigh]) + watcher_span
        watcher_freq = np.linspace(llim, ulim, 1000)
    
    # get the initial impedances and current etc at the first probe position
    Ztarg = target.lekid.total_impedance(carrier_freq_timestream[0])
    Zneigh = neighbour.lekid.total_impedance(carrier_freq_timestream[0])
    Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg + 1./Zneigh)
    Iin = carrier_Vin_timestream[0] / (Zpar + r2)
    Itarg = Iin * Zpar/Ztarg
    Ineigh = Iin * Zpar/Zneigh

    targ_ZL = 2.j*np.pi*(target.lekid.Lk + target.Lg)*carrier_freq_timestream[0]
    targ_ZRLC = target.lekid.parallel_RLC(carrier_freq_timestream[0])
    targ_IL = Itarg * targ_ZRLC / (targ_ZL + target.lekid.R)

    neigh_ZL = 2.j*np.pi*(neighbour.lekid.Lk + neighbour.Lg)*carrier_freq_timestream[0]
    neigh_ZRLC = neighbour.lekid.parallel_RLC(carrier_freq_timestream[0])
    neigh_IL = Itarg * neigh_ZRLC / (neigh_ZL + target.lekid.R)

    outdict = {}
    params = ['Lk', 'Lk_Isq', 'R', 'fr', 'IL', 'Ires', 'Zres', 'nqp']
    carrier_params = ['carrier_freq', 'carrier_Vin', 'carrier_Vout', 'Iin']
    
    if targ_nqp_timestream is None:
        targ_nqp_timestream = np.ones(len(carrier_Vin_timestream)) * target.calc_nqp()
    if neigh_nqp_timestream is None:
        neigh_nqp_timestream = np.ones(len(carrier_Vin_timestream)) * neighbour.calc_nqp()

    stop_flag = False

    # start sweepin'
    for step in range(len(carrier_freq_timestream)):
        outdict[step] = {}
        if step == 0:
            outdict[step]['target'] = target
            outdict[step]['neighbour'] = neighbour
            if save_watcher_data:
                outdict[0]['watcher_freq'] = watcher_freq # just save this once since it's always the same
                
        carrier_freq = carrier_freq_timestream[step]
        carrier_Vin = carrier_Vin_timestream[step]
        outdict[step]['carrier_freq'] = carrier_freq
        outdict[step]['carrier_Vin'] = carrier_Vin
        outdict[step]['carrier_Vout'] = {}
        outdict[step]['carrier_Vout']['iters'] = []
        outdict[step]['Iin'] = {}
        outdict[step]['Iin']['iters'] = []
    
        for resonator in ['targ', 'neigh']:
            outdict[step][resonator] = {}
            for param in params:
                outdict[step][resonator][param] = {}
                outdict[step][resonator][param]['iters'] = []
        
        # get the base impedances due to the qp population        
        # targ_nqp = target.calc_nqp()
        targ_nqp = targ_nqp_timestream[step]
        targ_sigma1 = target.calc_sigma1(f=carrier_freq, nqp=targ_nqp)
        targ_sigma2 = target.calc_sigma2(f=carrier_freq, nqp=targ_nqp)
        targ_sigma = targ_sigma1 - 1.j*targ_sigma2

        # neigh_nqp = neighbour.calc_nqp()
        neigh_nqp = neigh_nqp_timestream[step]
        neigh_sigma1 = neighbour.calc_sigma1(f=carrier_freq, nqp=neigh_nqp)
        neigh_sigma2 = neighbour.calc_sigma2(f=carrier_freq, nqp=neigh_nqp)
        neigh_sigma = neigh_sigma1 - 1.j*neigh_sigma2

        
        
        # up to this point these should always be the same on each iter, since we are not changing nqp.
        for iiter in range(niters): # iterate at this carrier freq and Vin to steady state\
            outdict[step]['Iin']['iters'].append(Iin)

            outdict[step]['targ']['nqp']['iters'].append(targ_nqp)
            outdict[step]['neigh']['nqp']['iters'].append(neigh_nqp)
            
            outdict[step]['targ']['Zres']['iters'].append(Ztarg)
            outdict[step]['neigh']['Zres']['iters'].append(Zneigh)
            
            outdict[step]['targ']['Ires']['iters'].append(Itarg)
            outdict[step]['neigh']['Ires']['iters'].append(Ineigh)
            
            outdict[step]['targ']['IL']['iters'].append(targ_IL)
            outdict[step]['neigh']['IL']['iters'].append(neigh_IL)
            
            # compute the base impedance at this frequency
            targ_Zs = target.calc_Zs(f=carrier_freq, sigma=targ_sigma) 
            targ_R, targ_Lk = target.calc_R_L(f=carrier_freq, Zs=targ_Zs)
            targ_R = targ_R + target.R_spoiler
            outdict[step]['targ']['Lk']['iters'].append(targ_Lk)
            outdict[step]['targ']['R']['iters'].append(targ_R)

            neigh_Zs = neighbour.calc_Zs(f=carrier_freq, sigma=neigh_sigma) 
            neigh_R, neigh_Lk = neighbour.calc_R_L(f=carrier_freq, Zs=neigh_Zs)
            neighR = neigh_R + neighbour.R_spoiler
            outdict[step]['neigh']['Lk']['iters'].append(neigh_Lk)
            outdict[step]['neigh']['R']['iters'].append(neigh_R)
            
            # add the nonlinear current response
            targ_Lk = targ_Lk * (1. + abs(targ_IL)**2 / Icrit**2)  
            neigh_Lk = neigh_Lk * (1. + abs(neigh_IL)**2 / Icrit**2)  
            
            outdict[step]['targ']['Lk_Isq']['iters'].append(targ_Lk)
            outdict[step]['neigh']['Lk_Isq']['iters'].append(neigh_Lk)
            
            # generate new lekid with these params
            targ_lekid_params = dict(R=targ_R, Lk=targ_Lk, Lg=target.Lg, C=target.C, Cc=target.Cc, Vin=carrier_Vin)
            targ_new_lekid = MR_LEKID(**targ_lekid_params)
            
            neigh_lekid_params = dict(R=neigh_R, Lk=neigh_Lk, Lg=neighbour.Lg, C=neighbour.C, Cc=neighbour.Cc, Vin=carrier_Vin)
            neigh_new_lekid = MR_LEKID(**neigh_lekid_params)
            
            # compute impedances and current for next iter
            Ztarg = targ_new_lekid.total_impedance(fc=carrier_freq)
            Zneigh = neigh_new_lekid.total_impedance(fc=carrier_freq)
            Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg + 1./Zneigh)
            Iin = carrier_Vin / (Zpar + r2)
            
            next_Itarg = Iin * Zpar/Ztarg
            next_Ineigh = Iin * Zpar/Zneigh
            Itarg = Itarg + damp*(next_Itarg - Itarg)
            Ineigh = Ineigh + damp*(next_Ineigh - Ineigh)
#             Ires_fs = nominal_tau_targ*2.1 # some fast "nature" sampling rate
#             ttt = 1./Ires_fs
#             Itarg = next_Itarg + (Itarg - next_Itarg) * np.exp(-ttt / nominal_tau_targ)
#             Ineigh = next_Ineigh + (Ineigh - next_Ineigh) * np.exp(-ttt / nominal_tau_neigh)
            
            targ_ZL = 2.j*np.pi*(targ_Lk + target.Lg)*carrier_freq
            targ_ZRLC = targ_new_lekid.parallel_RLC(carrier_freq)
            targ_IL = Itarg * targ_ZRLC / (targ_ZL + targ_R)
            targ_fr = targ_new_lekid.compute_fr()
        
            neigh_ZL = 2.j*np.pi*(neigh_Lk + neighbour.Lg)*carrier_freq
            neigh_ZRLC = neigh_new_lekid.parallel_RLC(carrier_freq)
            neigh_IL = Ineigh * neigh_ZRLC / (neigh_ZL + neigh_R)
            neigh_fr = neigh_new_lekid.compute_fr()
            
            outdict[step]['targ']['fr']['iters'].append(targ_fr)
            outdict[step]['neigh']['fr']['iters'].append(neigh_fr)
            
            Vout = carrier_Vin * (Zpar / (r2 + Zpar))
            outdict[step]['carrier_Vout']['iters'].append(Vout)
            
            
        # get the whole transfer function from the "watcher" measurement
        # only do this after the iterations or else the output is tooooo big
        if save_watcher_data:
            Ztarg = targ_new_lekid.total_impedance(fc=watcher_freq)
            Zneigh = neigh_new_lekid.total_impedance(fc=watcher_freq)
            Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg + 1./Zneigh)
            watcher_Vout = 1. * (Zpar / ( r2 + Zpar))
            outdict[step]['watcher_Vout'] = watcher_Vout
            
        for resonator in ['targ', 'neigh']:    
            for param in params:
                outdict[step][resonator][param]['final'] = outdict[step][resonator][param]['iters'][-1]
        outdict[step]['carrier_Vout']['final'] = outdict[step]['carrier_Vout']['iters'][-1]
        outdict[step]['Iin']['final'] = outdict[step]['Iin']['iters'][-1]
        if stop_when_comb_hithered:
            # fdetune = fc - fr --> fr = fc - fdetune
            # print('%.3e'%(abs((carrier_freq - comb_hither_fdetune) - targ_fr)))
            if abs((carrier_freq - comb_hither_fdetune) - targ_fr) < comb_hither_threshold:
            # if abs((carrier_freq - targ_fr) - comb_hither_fdetune) < comb_hither_threshold:
                stop_flag = True
                # print('Comb hither! stopping at step %d'%step)

        if stop_flag:
            break

    # outdict[step]['Iin']['final'] = outdict[step]['Iin']['iters'][-1]
    # outdict[step]['carrier_Vout']['final'] = outdict[step]['carrier_Vout']['iters'][-1]
    target.lekid = targ_new_lekid
    neighbour.lekid = neigh_new_lekid
    outdict[step]['target'] = target
    outdict[step]['neighbour'] = neighbour
    outdict[step]['comb_hithered'] = stop_flag

    return outdict


    





def set_up_for_feedback(res, foffset=-30e3, 
                      carrier_Vin_timestream=np.linspace(10e-5, 50e-6, 100),
                        fdetune=0, comb_hither_threshold=50., 
                        gather_calsamps=True, make_plots=False):
    

    fr0 = res.lekid.compute_fr()
    carrier_freq = res.lekid.compute_fr() + foffset + fdetune
    carrier_freq_timestream = np.ones(len(carrier_Vin_timestream)) * carrier_freq

    # first just sweep across the whole range and get the comb hither point
    exploratory_outdict = single_resonance_iterator(res=res, carrier_freq_timestream=carrier_freq_timestream, 
                                                 carrier_Vin_timestream=carrier_Vin_timestream,
                                                nqp_timestream=None,
                                                niters=200, damp=0.1,
                                                stop_when_comb_hithered=False, comb_hither_fdetune=fdetune,
                                                 comb_hither_threshold=comb_hither_threshold)
    
    steps = list(exploratory_outdict.keys())

    carrier_Vin = np.asarray([exploratory_outdict[step]['carrier_Vin'] for step in steps])
    fr = np.asarray([exploratory_outdict[step]['targ']['fr']['final'] for step in steps])
    fsep= 1e-3*(carrier_freq - (fr))
    # interpolate the sweep  since the comb hither point probably wasn't dead on
    fsep_interp = interpolate.interp1d(carrier_Vin, fsep)
    Vin_interp = np.linspace(carrier_Vin[0], carrier_Vin[-1], 10000)
    comb_hither_ind = abs(fsep_interp(Vin_interp) - 1e-3*fdetune).argmin()
    comb_hither_Vin = Vin_interp[comb_hither_ind]
#     print(min(abs(fsep_interp(Vin_interp))))
    
    
    if make_plots:
        nplots = 3
        nperrow = 3
        nrows = 1
        fig = plt.figure(figsize=(6*nperrow, 5*nrows))
        axarr = []
        for x in range(nplots):
            ax = fig.add_subplot(nrows, nperrow, x+1)
            axarr.append(ax)
        fig.suptitle('foffset %.1e, fdetune %.1e'%(foffset, fdetune))
        axarr[2].plot(carrier_Vin, fsep, label='exploratory')
        axarr[2].scatter(comb_hither_Vin, fsep_interp(comb_hither_Vin), marker='o')
    
    # now steady state iterate with a voltage itmestream going to this point
    if abs(foffset) > 20e3:
        nsteps = len(carrier_Vin_timestream) * 3
    else:
        nsteps = len(carrier_Vin_timestream) * 2
    targeted_Vin_timestream = np.logspace(np.log10(carrier_Vin[0]), 
        np.log10(comb_hither_Vin), 
        nsteps)
    carrier_freq_timestream = np.ones(len(targeted_Vin_timestream)) * carrier_freq
    outdict = single_resonance_iterator(res=res, carrier_freq_timestream=carrier_freq_timestream, 
                                                 carrier_Vin_timestream=targeted_Vin_timestream,
                                                nqp_timestream=None,
                                                niters=200, damp=0.1,
                                                stop_when_comb_hithered=True,
                                                comb_hither_fdetune=fdetune,
                                                 comb_hither_threshold=comb_hither_threshold)
    steps = list(outdict.keys())
    # print(steps, steps[-1])

    
    # use another steady state iterator to evaluate the quadrature of the initialized resonator's response
    comb_hithered_resonator = outdict[steps[-1]]['resonator']
    Vout_setpoint = outdict[steps[-1]]['carrier_Vout']['final']
    Vin_setpoint = outdict[steps[-1]]['carrier_Vin']
    watcher_Vout = outdict[steps[-1]]['watcher_Vout']
    nqp = comb_hithered_resonator.calc_nqp()
    nqp_timestream = np.random.normal(nqp, nqp*0.001, 30)
    print(np.mean(nqp_timestream), np.std(nqp_timestream))
    carrier_Vin_timestream = Vin_setpoint * np.ones(len(nqp_timestream))
    carrier_freq_timestream = carrier_freq * np.ones(len(nqp_timestream))
    outdict_calsamps = single_resonance_iterator(res=comb_hithered_resonator, 
                                                 carrier_freq_timestream=carrier_freq_timestream, 
                                                 carrier_Vin_timestream=carrier_Vin_timestream,
                                                nqp_timestream=nqp_timestream)
    

    
    if make_plots:
        carrier_freq = outdict[steps[0]]['carrier_freq']
        watcher_freq = outdict[0]['watcher_freq']

        # print(len(steps))
        if len(steps) > 1:
            skip = len(steps) // 10
        else:
            skip = 1
        for step in steps[::skip]:
            plotcolour = plt.cm.gnuplot(float(step) / len(steps))
            watcher_Vout = outdict[step]['watcher_Vout']
            mag_db = 20*np.log10(abs(watcher_Vout)/abs(watcher_Vout[-1]))
            axarr[0].plot(1e-3*(watcher_freq-fr0), mag_db, alpha=0.7, color=plotcolour)
        # always plot the last one
        watcher_Vout = outdict[steps[-1]]['watcher_Vout']
        mag_db = 20*np.log10(abs(watcher_Vout)/abs(watcher_Vout[-1]))
        axarr[0].plot(1e-3*(watcher_freq-fr0), mag_db, alpha=0.7, color='tab:red')
        axarr[0].axvline(1e-3*(carrier_freq-fr0), label='$f_{carrier}$')
        axarr[0].set_xlabel('f - fr0 [kHz]')

        carrier_Vin = np.asarray([outdict[step]['carrier_Vin'] for step in steps])
        fr = np.asarray([outdict[step]['targ']['fr']['final'] for step in steps])
        fsep= 1e-3*(carrier_freq - fr)

        Lk_Isq = np.asarray([outdict[step]['targ']['Lk_Isq']['final'] for step in steps])
        Lk = np.asarray([outdict[step]['targ']['Lk']['final'] for step in steps])
        # nqp = np.asarray([outdict[step]['targ']['nqp'] for step in steps])
        # t = np.asarray([outdict[step]['t'] for step in steps])
        IL = np.asarray([outdict[step]['targ']['IL']['final'] for step in steps])
        Zres = np.asarray([outdict[step]['targ']['Zres']['final'] for step in steps])

        axarr[2].plot(carrier_Vin, fsep, '--', label='targeted')
        axarr[2].axhline(0, label='$f_{carrier}$')
        axarr[2].axhline(fdetune*1e-3, linestyle='--', label='$f_{offset}$')
        axarr[2].set_ylabel('$f_{carrier} - f_r$ [kHz]')

        # axarr[3].plot(steps, Lk, label='Lk')
        # axarr[3].plot(steps, Lk_Isq, label='Lk(I)')
        # axarr[3].set_ylabel('Lk')

        # axarr[4].plot(steps, abs(IL))
        # axarr[4].set_ylabel('IL')

        # axarr[5].plot(steps, abs(Zres))
        # axarr[5].set_ylabel('Zres')

    # scatter the calsamps on top
    calsamps_steps = list(outdict_calsamps.keys())
    carrier_Vin = np.asarray([outdict_calsamps[step]['carrier_Vin'] for step in calsamps_steps])
    carrier_Vout = np.asarray([outdict_calsamps[step]['carrier_Vout']['final'] for step in calsamps_steps])
    if make_plots:
        axarr[1].scatter(carrier_Vout.real, carrier_Vout.imag, alpha=0.7)
        axarr[1].scatter(Vout_setpoint.real, Vout_setpoint.imag, marker='*', s=250, label='setpoint')
        axarr[1].set_ylabel('calsamps Q')
        axarr[1].set_ylabel('calsamps I')
        utils.square_axes(axarr[1])

    calsamps = carrier_Vout
        
    if make_plots:
        for x in axarr:
            x.legend()
        for x in axarr[2:]:
            x.set_xlabel('sample #')
        axarr[2].set_xlabel('Vin')
        fig.tight_layout()

        

    return outdict[steps[-1]], calsamps



