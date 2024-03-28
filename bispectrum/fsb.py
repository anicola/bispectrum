import numpy as np 
import pyccl as ccl
from functools import partial
import logging
import sys
from pyshtools.utils import Wigner3j
import bispectrum as bs
from scipy.interpolate import RegularGridInterpolator

class FSB(object):

    def __init__(self):
        pass

    def precompute_w3j(self, lmax):
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

    def Bk_fsb_interp(self, cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, ls_bins, ll_bins, Bkm='tree', nbar=None):
        '''
        Compute the FSB with interpolation.
        params:
        cosmo: ccl.Cosmology object
        tr1, tr2, tr3: ccl.Tracer objects
        ptt1, ptt2, ptt3: ccl.PTTracer objects
        ls_bins: filered ell range
        ll_bins: full ell range
        Bkm: bispectrum theory model, can be 'tree' or 'bihalofit'
        nbar: number density of the tracers, if not None, shot noise will be included
        '''

        lmin = np.min([np.min(ls_bins), np.min(ll_bins)])
        lmax = np.max([np.max(ls_bins), np.max(ll_bins)])

        if not hasattr(self, 'bl1l2l3_interp'):
            l1 = l2 = l3 = np.concatenate((np.arange(lmin, 10).astype('int'), np.linspace(10, 96, 20).astype('int'), 
                                           np.unique(np.geomspace(100, 767, 50).astype('int'))))
            bl1l2l3 = bs.Bl_ev(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm=Bkm)
            self.bl1l2l3_interp = RegularGridInterpolator((np.log(l1), np.log(l2), np.log(l3)), 
                                                          np.log(bl1l2l3))

            self.big_w3j = self.precompute_w3j(lmax)
        
        l3 = np.arange(lmin, lmax)
        l1 = l2 = np.arange(ls_bins[0], ls_bins[-1])
        l_grid = np.meshgrid(l1, l2, l3, indexing='ij')
        l_list = np.log(np.reshape(l_grid, (3, -1), order='C')).T

        ll_lower = ll_bins[:-1]
        ll_upper = ll_bins[1:]
        nbin_ll = ll_lower.shape[0]

        bl1l2l3 = np.exp(self.bl1l2l3_interp(l_list)).reshape(l1.shape[0], l2.shape[0], l3.shape[0])

        pt_types = [ptt1.type, ptt2.type, ptt3.type]

        # All tracers are number counts
        if pt_types.count('NC') == 3:
            if ptt1 == ptt2 == ptt3:
                # Compute bias
                b = ptt1.b1(0)
                if nbar is not None:
                    # Compute cls
                    cl = ccl.angular_cl(cosmo, tr1, tr2, l3)

                    bl1l2l3 = bl1l2l3 + 1./nbar**2 + \
                        1./nbar*b**2*(cl[ls_bins[0]-lmin:ls_bins[-1]-lmin, np.newaxis, np.newaxis] + \
                        cl[np.newaxis, ls_bins[0]-lmin:ls_bins[-1]-lmin, np.newaxis] + \
                        cl[np.newaxis, np.newaxis, :])
                else:
                    bl1l2l3 = bl1l2l3

            else:
                raise Exception('Shot noise is not implemented for cross-correlation of tracers')
            
        # Two tracers are number counts
        elif pt_types.count('NC') == 2:

            pg = []
            pm = []
            trg = []
            trm = []

            if ptt1.type == 'NC':
                pg.append(ptt1)
                trg.append(tr1)
            else:
                pm.append(ptt1)
                trm.append(tr1)
            if ptt2.type == 'NC':
                pg.append(ptt2)
                trg.append(tr2)
            else:   
                pm.append(ptt2)
                trm.append(tr2)
            if ptt3.type == 'NC':  
                pg.append(ptt3)
                trg.append(tr3)
            else:   
                pm.append(ptt3)
                trm.append(tr3)  

            cl = ccl.angular_cl_tracer_tracer(cosmo, trg[0], trm[0], l3)

            # Compute bias
            b = pg[0].b1(0)
            if pg[0] == pg[1] and nbar is not None:
                bl1l2l3 = bl1l2l3 + 1./nbar*b*cl[np.newaxis, np.newaxis, :]
            else:
                bl1l2l3 = bl1l2l3

        Bl_fsb = (2*l1[:, np.newaxis, np.newaxis]+1)*(2*l2[np.newaxis, :, np.newaxis]+1)*\
                    (2*l3[np.newaxis, np.newaxis, :]+1)/4./np.pi*\
                    self.big_w3j[ls_bins[0]:ls_bins[-1], ls_bins[0]:ls_bins[-1], lmin:]*bl1l2l3

        Bl_fsb_binned = np.zeros(nbin_ll)
        for ll in range(nbin_ll):
            Bl_fsb_binned[ll] = np.sum(Bl_fsb[:, :, ll_lower[ll]-lmin:ll_upper[ll]-lmin])/\
            np.sum((2*l3[np.newaxis, np.newaxis, ll_lower[ll]-lmin:ll_upper[ll]-lmin]+1))
        
        return Bl_fsb_binned, bl1l2l3


    def Bk_fsb(self, cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, ls_bins, ll_bins, Bkm='tree', nbar=None):
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

        big_w3j = self.precompute_w3j(lmax)

        bl1l2l3 = bs.Bl_ev(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm=Bkm)

        pt_types = [ptt1.type, ptt2.type, ptt3.type]

        # All tracers are number counts
        if pt_types.count('NC') == 3:
            if ptt1 == ptt2 == ptt3:
                # Compute bias
                b = ptt1.b1(0)
                if nbar is not None:
                    # Compute cls
                    cl = ccl.angular_cl(cosmo, tr1, tr2, l3)

                    bl1l2l3 = bl1l2l3 + 1./nbar**2 + \
                        1./nbar*b**2*(cl[ls_bins[0]:ls_bins[-1], np.newaxis, np.newaxis] + \
                        cl[np.newaxis, ls_bins[0]:ls_bins[-1], np.newaxis] + \
                        cl[np.newaxis, np.newaxis, :])
                else:
                    bl1l2l3 = bl1l2l3

            else:
                raise Exception('Shot noise is not implemented for cross-correlation of tracers')
            
        # Two tracers are number counts
        elif pt_types.count('NC') == 2:

            pg = []
            pm = []
            trg = []
            trm = []

            if ptt1.type == 'NC':
                pg.append(ptt1)
                trg.append(tr1)
            else:
                pm.append(ptt1)
                trm.append(tr1)
            if ptt2.type == 'NC':
                pg.append(ptt2)
                trg.append(tr2)
            else:   
                pm.append(ptt2)
                trm.append(tr2)
            if ptt3.type == 'NC':  
                pg.append(ptt3)
                trg.append(tr3)
            else:   
                pm.append(ptt3)
                trm.append(tr3)  

            cl = ccl.angular_cl_tracer_tracer(cosmo, trg[0], trm[0], l3)

            # Compute bias
            b = pg[0].b1(0)
            if pg[0] == pg[1] and nbar is not None:
                bl1l2l3 = bl1l2l3 + 1./nbar*b*cl[np.newaxis, np.newaxis, :]
            else:
                bl1l2l3 = bl1l2l3

        Bl_fsb = (2*l1[:, np.newaxis, np.newaxis]+1)*(2*l2[np.newaxis, :, np.newaxis]+1)*\
                    (2*l3[np.newaxis, np.newaxis, :]+1)/4./np.pi*\
                    big_w3j[ls_bins[0]:ls_bins[-1], ls_bins[0]:ls_bins[-1], :]*bl1l2l3

        Bl_fsb_binned = np.zeros(nbin_ll)
        for ll in range(nbin_ll):
            Bl_fsb_binned[ll] = np.sum(Bl_fsb[:, :, ll_lower[ll]:ll_upper[ll]])/\
            np.sum((2*l3[np.newaxis, np.newaxis, ll_lower[ll]:ll_upper[ll]]+1))
        
        return Bl_fsb_binned, bl1l2l3