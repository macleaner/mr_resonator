'''
iteratively solve the impedances of a resonance in steady state

I^2 prioritized


maclean.rouble@mail.mcgill.ca
'''


import numpy as np

from mr_complex_resonator import MR_complex_resonator as MR_complex_resonator
from mr_lekid import MR_LEKID as MR_LEKID
import utils


def two_resonance_iterator(target, neighbour, carrier_freq_timestream, carrier_Vin_timestream,
                    save_watcher_data=True, watcher_span=300e3, Icrit=1e-3,
                    niters=150, damp=0.1,
                    verbose=False):
    '''
    
    '''


    # probe sweep parameters
    fr = target.lekid.compute_fr()
    fneigh = neighbour.lekid.compute_fr()
    
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

    outdict = {}
    params = ['Lk', 'Lk_Isq', 'R', 'fr', 'IL', 'Ires', 'Zres']
    carrier_params = ['carrier_freq', 'carrier_Vin', 'carrier_Vout', 'Iin']
    
    
    # start sweepin'
    for step in range(len(carrier_freq_timestream)):
        outdict[step] = {}
        if step == 0:
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
        targ_nqp = target.calc_nqp()
        targ_sigma1 = target.calc_sigma1(f=carrier_freq, nqp=targ_nqp)
        targ_sigma2 = target.calc_sigma2(f=carrier_freq, nqp=targ_nqp)
        targ_sigma = targ_sigma1 - 1.j*targ_sigma2

        neigh_nqp = neighbour.calc_nqp()
        neigh_sigma1 = neighbour.calc_sigma1(f=carrier_freq, nqp=neigh_nqp)
        neigh_sigma2 = neighbour.calc_sigma2(f=carrier_freq, nqp=neigh_nqp)
        neigh_sigma = neigh_sigma1 - 1.j*neigh_sigma2

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
        
        # get the initial impedances and current etc at this probe position
        Ztarg = target.lekid.total_impedance(fc=carrier_freq, L=targ_Lk+target.lekid.Lg, R=targ_R)
        Zneigh = neighbour.lekid.total_impedance(fc=carrier_freq, L=neigh_Lk+neighbour.lekid.Lg, R=neigh_R)
        Zpar = 1./(1./ZLNA + 1./r3 + 1./Ztarg + 1./Zneigh)
        Iin = carrier_Vin / (Zpar + r2)
        
        Itarg = Iin * Zpar/Ztarg
        Ineigh = Iin * Zpar/Zneigh
        
        targ_ZL = 2.j*np.pi*(targ_Lk + target.Lg)*carrier_freq
        targ_ZRLC = target.lekid.parallel_RLC(carrier_freq)
        targ_IL = Itarg * targ_ZRLC / (targ_ZL + targ_R)
        
        neigh_ZL = 2.j*np.pi*(neigh_Lk + neighbour.Lg)*carrier_freq
        neigh_ZRLC = neighbour.lekid.parallel_RLC(carrier_freq)
        neigh_IL = Itarg * neigh_ZRLC / (neigh_ZL + neigh_R)
        
        # up to this point these should always be the same on each iter, since we are not changing nqp.

        for iiter in range(niters): # iterate at this carrier freq and Vin to steady state\
            outdict[step]['Iin']['iters'].append(Iin)
            
            outdict[step]['targ']['Zres']['iters'].append(Ztarg)
            outdict[step]['neigh']['Zres']['iters'].append(Zneigh)
            
            outdict[step]['targ']['Ires']['iters'].append(Itarg)
            outdict[step]['neigh']['Ires']['iters'].append(Ineigh)
            
            outdict[step]['targ']['IL']['iters'].append(targ_IL)
            outdict[step]['neigh']['IL']['iters'].append(neigh_IL)
            
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
#             watcher_Vout = 1. * targ.lekid.ptype(Zpar, r1, r2, r3)
            watcher_Vout = 1. * (Zpar / ( r2 + Zpar))
            outdict[step]['watcher_Vout'] = watcher_Vout
            
        for resonator in ['targ', 'neigh']:    
            for param in params:
#                 print(param, step, outdict[step][resonator][param])
                outdict[step][resonator][param]['final'] = outdict[step][resonator][param]['iters'][-1]
#                 outdict[step][resonator][param]['iters'] = np.asarray(outdict[step][resonator][param]['iters'])

    outdict[step]['Iin']['final'] = outdict[step]['Iin']['iters'][-1]
    outdict[step]['carrier_Vout']['final'] = outdict[step]['carrier_Vout']['iters'][-1]
#     outdict[step]['Iin']['iters'] = np.asarray(outdict[step]['Iin']['iters'])
#     outdict[step]['carrier_Vout']['iters'] = np.asarray(outdict[step]['carrier_Vout']['iters'])
        
    return outdict


    
