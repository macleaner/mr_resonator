'''
iteratively solve the impedances of a resonance as a probe tone is swept across it

includes nonlinearities via goldie&withington's numerical thermal approximation, and I^2


maclean.rouble@mail.mcgill.ca
'''


import numpy as np

from mr_complex_resonator import MR_complex_resonator as MR_complex_resonator
from mr_lekid import MR_LEKID as MR_LEKID
import utils




def freq_sweep_iterator(res, sweep_frange, 
                    save_watcher_data=True, watcher_span=300e3, Icrit=1,
                    niters=150, damp=0.1, stability_tolerance=1e-4, 
                    verbose=False):
    '''
    
    '''


    # probe sweep parameters
    fr = res.lekid.compute_fr()
    # # the two directions for the sweep
    # if direction.lower() == 'up':
    #     sweep_frange = np.linspace(fr-sweep_span, fr+sweep_span/2, n_sweep_pts)
    # else:
    #     sweep_frange = np.linspace(fr+sweep_span/2, fr-sweep_span, n_sweep_pts)

    # if direction.lower() == 'DOWN':
    
    # optionally capture Vout across the whole resonance bandwidth, each time we move the probe
    ## this is analogous to a low-power sweep or a multifrequency snapshot
    ### so we call it a "watcher" measurement to distinguish from the actual
    ### probe frequency sweep that we are computing with the nonlinearity etc
    if save_watcher_data:
        # watcher_span = 300e3
        watcher_freq = np.linspace(fr-watcher_span, fr+watcher_span, 1000)
    
    
    
    # get the initial impedances and current etc at the first probe position
    Zres = res.lekid.total_impedance(sweep_frange[0])
    Vin = res.Vin
    probe_Vout = res.lekid.compute_Vout(sweep_frange[0])
    
    # we need to know the other impedances in the nearby circuit to compute the current
    r1, r2, r3 = res.lekid.get_att_vals(res.lekid.input_atten_dB)
    ZLNA = 50.
    Zatt = r3
    
    starting_Lk = res.lekid.Lk
    Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres )
    Iin = Vin / (Zpar + r2)
    Ires = Iin * Zpar/Zres
    
    
    # initial nonlinear effective temperature etc
    TN = res.T
    # T_stage = res.T
    # Tb = T_stage
    nqp = res.calc_nqp()
    
    outdict = {}
    params = ['Lk', 'Lk_Isq', 'R', 'fr', 'Ires', 'Iin', 'Zres', 
              'probe_Vout', 'probe_freq',
             'nqp', 'TN', 'TN_passed', 'Pabs']
    
    # start sweepin'
    for fstep, probe_freq in enumerate(sweep_frange):
        outdict[fstep] = {}
        if fstep == 0:
            if save_watcher_data:
                outdict[0]['watcher_freq'] = watcher_freq # just save this once since it's always the same
                
        for param in params:
            outdict[fstep][param] = {}
            outdict[fstep][param]['iters'] = []

        converged = False
        min_iters = 30
        for iiter in range(niters):
            TN, Ires, new_lekid, new_probe_Vout = iterator_inner(outdict=outdict, fstep=fstep, res=res, 
                                                   probe_freq=probe_freq, 
                                                   TN=TN, Ires=Ires, Icrit=Icrit, damp=damp,
                                                   ZLNA=ZLNA, r2=r2, r3=r3)
            # right now we are just checking if the output voltage is changing
            # but this could be expanded to check a set of variables for stability
            delta = (abs((abs(new_probe_Vout) - abs(probe_Vout))))/res.Vin
            
            if delta < stability_tolerance and iiter > min_iters:
                if verbose:
                    print('converged after %d iterations'%iiter)
                converged = True
                break
        if converged is False:
            if verbose:
                print('not converged (delta = %.2e'%delta)
        
            
        for param in params:
            outdict[fstep][param]['final'] = outdict[fstep][param]['iters'][-1]
            
        # get the whole transfer function from the "watcher" measurement
        if save_watcher_data:
            watcher_Vout = new_lekid.compute_Vout(watcher_freq)
            outdict[fstep]['watcher_Vout'] = watcher_Vout
        

    return outdict


    
def iterator_inner(outdict, fstep, res, damp,
                 probe_freq, TN, Ires, Icrit,
                ZLNA, r2, r3):
    outdict[fstep]['TN']['iters'].append(TN)
    nqp = res.calc_nqp(T=TN)
    outdict[fstep]['nqp']['iters'].append(nqp)

    sigma1 = res.calc_sigma1(f=probe_freq, nqp=nqp, T=TN)
    sigma2 = res.calc_sigma2(f=probe_freq, nqp=nqp, T=TN)
    sigma = sigma1 - 1.j*sigma2
    Zs = res.calc_Zs(f=probe_freq, sigma=sigma) 
    R, Lk = res.calc_R_L(f=probe_freq, Zs=Zs)
    R = R + res.R_spoiler 
    outdict[fstep]['Lk']['iters'].append(Lk)
    outdict[fstep]['R']['iters'].append(R)
    Lk = Lk * (1. + abs(Ires)**2 / Icrit**2)
    outdict[fstep]['Lk_Isq']['iters'].append(Lk)
    
    # generate new lekid with these params
    lekid_params = dict(R=R, Lk=Lk, Lg=res.Lg, C=res.C, Cc=res.Cc, Vin=res.Vin)
    new_lekid = MR_LEKID(**lekid_params)
    fr = new_lekid.compute_fr()
    outdict[fstep]['fr']['iters'].append(fr)
    # frs.append(new_lekid.compute_fr())

    Zres = new_lekid.total_impedance(probe_freq)
    outdict[fstep]['Zres']['iters'].append(Zres)

    Zpar = 1./(1./ZLNA + 1./r3 + 1./Zres)
    Iin = res.Vin / (Zpar + r2)
    next_Ires = Iin * Zpar/Zres
    Ires = Ires + damp*(next_Ires - Ires)
    outdict[fstep]['Ires']['iters'].append(Ires)
    outdict[fstep]['Iin']['iters'].append(Iin)

    probe_Vout = new_lekid.compute_Vout(probe_freq)
    outdict[fstep]['probe_freq']['iters'].append(probe_freq) # this doesn't change but whatever
    outdict[fstep]['probe_Vout']['iters'].append(probe_Vout)

    # convert absorbed power into a new effective temperature
    Pabs = res.calc_power_abs_in_res(fc=probe_freq, Zres=Zres, Ires=Ires)
    outdict[fstep]['Pabs']['iters'].append(Pabs)
    TN, passed = res.solve_for_Pabseta_at_T(Tguess=TN, true_Pabseta=Pabs, 
                                                 verbose=False, Tb=res.T)
    outdict[fstep]['TN']['iters'].append(TN)
    outdict[fstep]['TN_passed']['iters'].append(passed)
    return TN, Ires, new_lekid, probe_Vout