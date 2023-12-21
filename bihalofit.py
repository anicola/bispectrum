import numpy as np 
import pyccl as ccl
from functools import partial
import sys
import bispectrum as bs
import scipy.interpolate
from numba import jit

class Bihalofit(object):

    def __init__(self, cosmo):

        self.cosmo = cosmo
        a_arr = 1./(1+np.linspace(0, 5, 500))
        knl, neff = self.initialize_halofit(cosmo, a_arr)

        self.knl = scipy.interpolate.interp1d(a_arr, knl)
        self.neff = scipy.interpolate.interp1d(a_arr, neff)

    def sigma2R(self, cosmo, a, R, k_arr):

        pk = ccl.linear_matter_power(cosmo, k_arr, a=a)[:, np.newaxis]

        k_arr = k_arr[:, np.newaxis]
        R = R[np.newaxis, :]

        integ = k_arr**2 * pk * np.exp(-k_arr**2 * R**2)

        sigma2R = np.trapz(integ, k_arr, axis=0) / (2*np.pi**2)

        return sigma2R

    def dsigma2RdR(self, cosmo, a, R, k_arr):

        pk = ccl.linear_matter_power(cosmo, k_arr, a=a)

        integ = k_arr**4 * pk * np.exp(-k_arr**2 * R**2)

        dsigma2RdR = np.trapz(integ, k_arr, axis=0) / (2*np.pi**2)
        
        return dsigma2RdR

    def initialize_halofit(self, cosmo, a):

        # k sampling
        log10k_min=-4
        log10k_max=3
        nk_per_decade=50
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        k_arr = np.logspace(log10k_min, log10k_max, nk_total)

        R = 1./np.logspace(-4, 2, 5000)

        knl = np.zeros_like(a)
        neff = np.zeros_like(a)
        for i, ai in enumerate(a):

            sigma2R_temp = self.sigma2R(cosmo, ai, R, k_arr)
            rsigma = np.interp(1, sigma2R_temp, R)
            sigma_knl = self.sigma2R(cosmo, ai, np.array([rsigma]), k_arr)
            assert np.allclose(sigma_knl, 1., rtol=1e-4)
            knl[i] = 1./rsigma

            dsigma2RdR_temp = self.dsigma2RdR(cosmo, ai, 1./knl[i], k_arr)    
            neff[i] = -3 + 2*rsigma**2*dsigma2RdR_temp / sigma_knl

        return knl, neff

    # @jit()
    def B1h(self, k1, k2, k3, a, knl, neff, logsig8z):

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            q1 = k1/knl[np.newaxis, :]
            q2 = k2/knl[np.newaxis, :]
            q3 = k3/knl[np.newaxis, :]

            chi_arr = ccl.comoving_radial_distance(self.cosmo, a)
            k1_mesh = np.meshgrid(np.array((k1*chi_arr)[0]), (k2*chi_arr)[:, 0], (k3*chi_arr)[:, 0], indexing='ij')
            ksorted = np.sort(np.stack((k1_mesh[0][0, :, :], k1_mesh[1][0, :, :], k1_mesh[2][0, :, :])), axis=0)
            kmin = ksorted[0]
            kmid = ksorted[1]
            kmax = ksorted[-1]
            r1 = (kmin/kmax)[np.newaxis, :, :, np.newaxis]
            r2 = ((kmid+kmin-kmax)/kmax)[np.newaxis, :, :, np.newaxis]

            neff = neff[np.newaxis, np.newaxis, np.newaxis, :]
            logsig8z = logsig8z[np.newaxis, np.newaxis, np.newaxis, :]

            q1 = q1[:, np.newaxis, np.newaxis, :]
            q2 = q2[np.newaxis, :, np.newaxis, :]
            q3 = q3[np.newaxis, np.newaxis, :, :]

        else:
            q1 = k1/knl
            q2 = k2/knl 
            q3 = k3/knl

            ksorted = np.sort([k1, k2, k3], axis=0)
            kmin = ksorted[0]
            kmid = ksorted[1]
            kmax = ksorted[-1]
            r1 = kmin/kmax
            r2 = (kmid+kmin-kmax)/kmax

        gamman = 10**(0.182+0.57*neff)
        an = 10**(-2.167-2.944*logsig8z-1.106*logsig8z**2-2.865*logsig8z**3-0.310*r1**gamman)*self.cosmo['h']**2
        bn = 10**(-3.428 - 2.681*logsig8z + 1.624*logsig8z**2 - 0.095*logsig8z**3)*self.cosmo['h']**2
        cn = 10**(0.159 - 1.107*neff)
        alphan = np.minimum(10**(-4.348 - 3.006*neff - 0.5745*neff**2 + 10**(-0.9+0.2*neff)*r2**2), 1 - 2./3*self.cosmo['n_s'])
        betan = 10**(-1.731-2.845*neff-1.4995*neff**2-0.2811*neff**3+0.007*r2)

        B1h = (1./(an*q1**alphan + bn*q1**betan)) * (1./(1. + (cn*q1)**-1)) * \
                (1./(an*q2**alphan + bn*q2**betan)) * (1./(1. + (cn*q2)**-1)) * \
                (1./(an*q3**alphan + bn*q3**betan)) *  (1./(1. + (cn*q3)**-1)) 

        return B1h

    def I(self, q, en):

        I = 1./(1. + en*q)

        return I

    def PE(self, q, Pklin, fn, gn, hn, mn, mun, nn, nun, pn):

        PE = (1.+fn*q**2)/(1.+gn*q+hn*q**2)*Pklin + 1./(mn*q**mun+nn*q**nun)*1./(1.+(pn*q)**-3)

        return PE

    # @jit()
    def B3h(self, k1, k2, k3, a, Pk2d, knl, neff, logsig8z, Omegamz):

        fn = 10**(-10.533-16.838*neff-9.3048*neff**2-1.8263*neff**3)
        gn = 10**(2.787 + 2.405*neff+0.4577*neff**2)
        hn = 10**(-1.118-0.394*neff)
        mn = 10**(-2.605-2.434*logsig8z+5.71*logsig8z**2)*self.cosmo['h']**3       
        nn = 10**(-4.468-3.08*logsig8z+1.035*logsig8z**2)*self.cosmo['h']**3
        mun = 10**(15.312+22.977*neff+10.9579*neff**2+1.6586*neff**3)
        nun = 10**(1.347+1.246*neff+0.4525*neff**2)
        pn = 10**(0.071-0.433*neff)
        dn = 10**(-0.483+0.892*logsig8z-0.086*Omegamz)
        en = 10**(-0.632+0.646*neff)

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            q1 = (k1/knl[np.newaxis, :])[:, np.newaxis, np.newaxis, :] 
            q2 = (k2/knl[np.newaxis, :])[np.newaxis, :, np.newaxis, :]
            q3 = (k3/knl[np.newaxis, :])[np.newaxis, np.newaxis, :, :]
        else:
            q1 = k1/knl
            q2 = k2/knl 
            q3 = k3/knl

        Ik1 = self.I(q1, en)
        Ik2 = self.I(q2, en)
        Ik3 = self.I(q3, en)

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            Pklin1 = np.diag(Pk2d(k=k1, a=a))[np.newaxis, np.newaxis, np.newaxis, :] 
            Pklin2 = np.array([np.diag(Pk2d(k=k2[i, :], a=a)) for i in range(k2.shape[0])])[np.newaxis, :, np.newaxis, :]
            Pklin3 = np.array([np.diag(Pk2d(k=k3[i, :], a=a)) for i in range(k3.shape[0])])[np.newaxis, np.newaxis, :, :] 
        else:
            if len(a) == 1:
                Pklin1 = Pk2d(k=k1, a=a)
                Pklin2 = Pk2d(k=k2, a=a)
                Pklin3 = Pk2d(k=k3, a=a)
            else:
                Pklin1 = np.diag(Pk2d(k=k1, a=a))
                Pklin2 = np.diag(Pk2d(k=k2, a=a))
                Pklin3 = np.diag(Pk2d(k=k3, a=a))

        PE1 = self.PE(q1, Pklin1, fn, gn, hn, mn, mun, nn, nun, pn)
        PE2 = self.PE(q2, Pklin2, fn, gn, hn, mn, mun, nn, nun, pn)
        PE3 = self.PE(q3, Pklin3, fn, gn, hn, mn, mun, nn, nun, pn)

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            k1 = k1[np.newaxis, np.newaxis, np.newaxis, :]
            k2 = k2[np.newaxis, :, np.newaxis, :]
            k3 = k3[np.newaxis, np.newaxis, :, :]

        B3h = 2*((bs.F2(k1, k2, k3) + dn*q3)*Ik1*Ik2*Ik3*PE1*PE2 + \
                (bs.F2(k2, k3, k1) + dn*q1)*Ik2*Ik3*Ik1*PE2*PE3 + \
                (bs.F2(k3, k1, k2) + dn*q2)*Ik3*Ik1*Ik2*PE3*PE1)

        return B3h

    def Bk(self, k1, k2, k3, a, Pk2d=None):

        if Pk2d is None:
            Pk2d = bs.init_Pk2d(self.cosmo)

        logsigma8z = np.log10(np.array([ccl.power.sigmaR(self.cosmo, 8./self.cosmo['h'], a=ai, p_of_k_a=Pk2d) for ai in a]))
        Omegamz = ccl.omega_x(self.cosmo, a, 'matter')

        knl = self.knl(a)
        neff = self.neff(a)

        B1 = self.B1h(k1, k2, k3, a, knl, neff, logsigma8z)  

        B3 = self.B3h(k1, k2, k3, a, Pk2d, knl, neff, logsigma8z, Omegamz)

        Bihft = B1 + B3

        return Bihft

