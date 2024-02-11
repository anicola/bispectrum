import numpy as np 
import pyccl as ccl
from functools import partial
import logging
import sys
from pyshtools.utils import Wigner3j
import bispectrum as bs

def precompute_w3j(lmax):
    ''' 
    Precompute the Wigner-3j symbols for the FSB calculation.
    params:
    lmax: maximum ell value
    '''
    
    ells_w3j = np.arange(0, lmax)
    w3j = np.zeros_like(ells_w3j, dtype=float)
    big_w3j = np.zeros((lmax, lmax, lmax))
    for ell1 in ells_w3j[1:]:
        for ell2 in ells_w3j[1:]:
            w3j_array, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
            w3j_array = w3j_array[:ellmax - ellmin + 1]
            # make the w3j_array the same shape as the w3j
            if len(w3j_array) < len(ells_w3j):
                reference = np.zeros(len(w3j))
                reference[:w3j_array.shape[0]] = w3j_array
                w3j_array = reference

            w3j_array = np.concatenate([w3j_array[-ellmin:],
                                        w3j_array[:-ellmin]])
            w3j_array = w3j_array[:len(ells_w3j)]
            w3j_array[:ellmin] = 0

            big_w3j[:, ell1, ell2] = w3j_array

    big_w3j = big_w3j**2

    return big_w3j

# def Bk_fsb(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, ls_bins, ll_bins, Bkm='tree'):

#     lmax = np.max([np.max(ls_bins), np.max(ll_bins)])
#     l1 = l2 = l3 = np.arange(0, lmax)

#     ls_lower = ls_bins[:-1]
#     ls_upper = ls_bins[1:]
#     ll_lower = ll_bins[:-1]
#     ll_upper = ll_bins[1:]
#     nbin_ls = ls_lower.shape[0]
#     nbin_ll = ll_lower.shape[0]

#     big_w3j = precompute_w3j(lmax)

#     bl1l2l3 = bs.Bl_ev(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm=Bkm)

#     Bl_fsb = (2*l1[:, np.newaxis, np.newaxis]+1)*(2*l2[np.newaxis, :, np.newaxis]+1)*\
#                 (2*l3[np.newaxis, np.newaxis, :]+1)/4./np.pi*big_w3j*bl1l2l3

#     Bl_fsb_binned = np.zeros((nbin_ls, nbin_ll))
#     for ls in range(nbin_ls):
#         for ll in range(nbin_ll):
#             print((Bl_fsb[ls_lower[ls]:ls_upper[ls], ls_lower[ls]:ls_upper[ls], ll_lower[ll]:ll_upper[ll]]).shape)
#             Bl_fsb_binned[ls, ll] = np.sum(Bl_fsb[ls_lower[ls]:ls_upper[ls], 
#                             ls_lower[ls]:ls_upper[ls], ll_lower[ll]:ll_upper[ll]]/(2*l3[np.newaxis, np.newaxis, ll_lower[ll]:ll_upper[ll]]+1))
    
#     return Bl_fsb_binned, bl1l2l3

def Bk_fsb(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, ls_bins, ll_bins, Bkm='tree', nbar=None, bias=None):
    '''
    Compute the FSB.
    params:
    cosmo: ccl.Cosmology object
    tr1, tr2, tr3: ccl.Tracer objects
    ptt1, ptt2, ptt3: ccl.PTTracer objects
    ls_bins: filered ell range
    ll_bins: full ell range
    Bkm: bispectrum theory model, can be 'tree' or 'bihalofit'
    nbar: number density of the tracers, if not None, shot noise will be included
    '''

    lmax = np.max([np.max(ls_bins), np.max(ll_bins)])
    l3 = np.arange(0, lmax)
    l1 = l2 = np.arange(ls_bins[0], ls_bins[-1])

    ll_lower = ll_bins[:-1]
    ll_upper = ll_bins[1:]
    nbin_ll = ll_lower.shape[0]

    big_w3j = precompute_w3j(lmax)

    bl1l2l3 = bs.Bl_ev(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm=Bkm)

    if nbar is not None:
        cl = ccl.angular_cl(cosmo, tr1, tr2, l3)
        if bias is not None:
            bl1l2l3 = bias**3*bl1l2l3 + 1./nbar**2 + 1./nbar*bias**2*(cl[ls_bins[0]:ls_bins[-1], np.newaxis, np.newaxis] + \
                cl[np.newaxis, ls_bins[0]:ls_bins[-1], np.newaxis] + \
                cl[np.newaxis, np.newaxis, :])
        else:
            bl1l2l3 = bl1l2l3 + 1./nbar**2 + 1./nbar*(cl[ls_bins[0]:ls_bins[-1], np.newaxis, np.newaxis] + \
                cl[np.newaxis, ls_bins[0]:ls_bins[-1], np.newaxis] + \
                cl[np.newaxis, np.newaxis, :])

    Bl_fsb = (2*l1[:, np.newaxis, np.newaxis]+1)*(2*l2[np.newaxis, :, np.newaxis]+1)*\
                (2*l3[np.newaxis, np.newaxis, :]+1)/4./np.pi*\
                big_w3j[ls_bins[0]:ls_bins[-1], ls_bins[0]:ls_bins[-1], :]*bl1l2l3

    Bl_fsb_binned = np.zeros(nbin_ll)
    for ll in range(nbin_ll):
        Bl_fsb_binned[ll] = np.sum(Bl_fsb[:, :, ll_lower[ll]:ll_upper[ll]])/\
        np.sum((2*l3[np.newaxis, np.newaxis, ll_lower[ll]:ll_upper[ll]]+1))
    
    return Bl_fsb_binned, bl1l2l3