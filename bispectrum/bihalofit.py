import numpy as np 
import pyccl as ccl
from functools import partial
import sys
import bispectrum as bs
import scipy.interpolate
from numba import jit

class Bihalofit(object):

    def __init__(self, cosmo, a=None, Pk2d=None):
        '''
        Initialize the bihalofit class.
        params:
        cosmo: CCL cosmology object
        a: array, scale factor
        Pk2d: CCL Pk2d power spectrum object
        '''

        self.cosmo = cosmo
        a_arr = 1./(1+np.linspace(0, 5, 100))
        knl, neff = self.initialize_halofit(cosmo, a_arr)

        self.knl = scipy.interpolate.interp1d(a_arr, knl)
        self.neff = scipy.interpolate.interp1d(a_arr, neff)

        if a is not None and Pk2d is not None:
            self.chi_arr = ccl.comoving_radial_distance(self.cosmo, a)

            knl = self.knl(a)
            neff = self.neff(a)

            R8 = 8./self.cosmo['h']
            logsigma8z_arr = np.log10(np.array([ccl.power.sigmaR(self.cosmo, R8, a=ai, p_of_k_a=Pk2d) for ai in a_arr]))
            logsigma8z_intp = scipy.interpolate.interp1d(a_arr, logsigma8z_arr)
            logsigma8z = logsigma8z_intp(a)
            Omegamz = ccl.omega_x(self.cosmo, a, 'matter')

            # B1h
            self.gamman = 10**(0.182+0.57*neff)
            self.bn = 10**(-3.428 - 2.681*logsigma8z + 1.624*logsigma8z**2 - 0.095*logsigma8z**3)*self.cosmo['h']**2
            self.cn = 10**(0.159 - 1.107*neff)

            # B3h
            self.fn = 10**(-10.533-16.838*neff-9.3048*neff**2-1.8263*neff**3)
            self.gn = 10**(2.787 + 2.405*neff+0.4577*neff**2)
            self.hn = 10**(-1.118-0.394*neff)
            self.mn = 10**(-2.605-2.434*logsigma8z+5.71*logsigma8z**2)*self.cosmo['h']**3       
            self.nn = 10**(-4.468-3.08*logsigma8z+1.035*logsigma8z**2)*self.cosmo['h']**3
            self.mun = 10**(15.312+22.977*neff+10.9579*neff**2+1.6586*neff**3)
            self.nun = 10**(1.347+1.246*neff+0.4525*neff**2)
            self.pn = 10**(0.071-0.433*neff)
            self.dn = 10**(-0.483+0.892*logsigma8z-0.086*Omegamz)
            self.en = 10**(-0.632+0.646*neff)

            self.knl_arr = knl
            self.neff_arr = neff
            self.logsigma8z = logsigma8z
            self.Omegamz = Omegamz


    def sigma2R(self, Pk2D_lin, a, R, k_arr):
        ''' 
        Compute the variance of the density field smoothed with a top-hat filter of radius R.
        params:
        cosmo: CCL cosmology object
        a: array, scale factor
        R: array, radius of the top-hat filter
        k_arr: array, wavenumbers
        '''

        pk = Pk2D_lin(k=k_arr, a=a)[:, np.newaxis]

        k_arr = k_arr[:, np.newaxis]
        R = R[np.newaxis, :]

        integ = k_arr**2 * pk * np.exp(-k_arr**2 * R**2)

        sigma2R = np.trapz(integ, k_arr, axis=0) / (2*np.pi**2)

        return sigma2R

    def dsigma2RdR(self, Pk2D_lin, a, R, k_arr):
        '''
        Compute the derivative of the variance of the density field smoothed with a top-hat filter of radius R.
        params:
        cosmo: CCL cosmology object
        a: array, scale factor
        R: array, radius of the top-hat filter
        k_arr: array, wavenumbers
        '''

        pk = Pk2D_lin(k=k_arr, a=a)

        integ = k_arr**4 * pk * np.exp(-k_arr**2 * R**2)

        dsigma2RdR = np.trapz(integ, k_arr, axis=0) / (2*np.pi**2)
        
        return dsigma2RdR

    def initialize_halofit(self, cosmo, a):
        '''
        Initialize quantities needed by bihalofit.
        params:
        cosmo: CCL cosmology object
        a: array, scale factor
        '''

        # k sampling
        log10k_min=-4
        log10k_max=3
        nk_per_decade=50
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        k_arr = np.logspace(log10k_min, log10k_max, nk_total)

        R = 1./np.logspace(-2, 2, 600)

        a_arr = ccl.get_pk_spline_a()
        pk = np.array([ccl.linear_matter_power(cosmo, k_arr, a=ai) for ai in a_arr])
        Pk2D_lin = ccl.Pk2D(a_arr=a_arr,
                lk_arr=np.log(k_arr),
                pk_arr=pk,
                is_logp=False)

        knl = np.zeros_like(a)
        neff = np.zeros_like(a)
        for i, ai in enumerate(a):

            sigma2R_temp = self.sigma2R(Pk2D_lin, ai, R, k_arr)
            rsigma = np.interp(1, sigma2R_temp, R)
            sigma_knl = self.sigma2R(Pk2D_lin, ai, np.array([rsigma]), k_arr)
            assert np.allclose(sigma_knl, 1., rtol=1e-4)
            knl[i] = 1./rsigma

            dsigma2RdR_temp = self.dsigma2RdR(Pk2D_lin, ai, 1./knl[i], k_arr)    
            neff[i] = -3 + 2*rsigma**2*dsigma2RdR_temp / sigma_knl

        return knl, neff

    def B1h(self, k1, k2, k3, a, knl, neff, logsigma8z, cache=False):
        '''
        Compute the 1-halo term of the bispectrum using the bihalofit fitting function.
        See: https://arxiv.org/abs/1911.07886.
        params:
        k1, k2, k3: arrays, wavenumbers
        a: array, scale factor
        knl: array, non-linear scale
        neff: array, effective spectral index
        logsigma8z: array, log10 of the variance of the density field smoothed with a top-hat filter of radius 8 Mpc/h
        '''

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:

            if not hasattr(self, 'chi_arr'):
                chi_arr = ccl.comoving_radial_distance(self.cosmo, a)
                k1_mesh = np.meshgrid(np.array((k1*chi_arr)[0]), (k2*chi_arr)[:, 0], 
                            (k3*chi_arr)[:, 0], indexing='ij')
            else:
                if cache:
                    if not hasattr(self, 'k2_chi'):
                        self.k2_chi = (k2*self.chi_arr)[:, 0]
                        self.k3_chi = (k3*self.chi_arr)[:, 0]
                        k2_chi = self.k2_chi
                        k3_chi = self.k3_chi
                    else:
                        k2_chi = self.k2_chi
                        k3_chi = self.k3_chi
                else:
                    k2_chi = (k2*self.chi_arr)[:, 0]
                    k3_chi = (k3*self.chi_arr)[:, 0]
                k1_mesh = np.meshgrid(np.array((k1*self.chi_arr)[0]), k2_chi, 
                            k3_chi, indexing='ij')
            ksorted = np.sort(np.stack((k1_mesh[0][0, :, :], k1_mesh[1][0, :, :], 
                        k1_mesh[2][0, :, :])), axis=0)
            kmin = ksorted[0]
            kmid = ksorted[1]
            kmax = ksorted[-1]
            r1 = (kmin/kmax)[np.newaxis, :, :, np.newaxis]
            r2 = ((kmid+kmin-kmax)/kmax)[np.newaxis, :, :, np.newaxis]

            q1 = (k1/knl[np.newaxis, :])[:, np.newaxis, np.newaxis, :]
            if cache:
                if not hasattr(self, 'q2'):
                    self.q2 = (k2/knl[np.newaxis, :])[np.newaxis, :, np.newaxis, :]
                    self.q3 = (k3/knl[np.newaxis, :])[np.newaxis, np.newaxis, :, :]
                    q2 = self.q2
                    q3 = self.q3
                else:
                    q2 = self.q2
                    q3 = self.q3
            else:
                q2 = (k2/knl[np.newaxis, :])[np.newaxis, :, np.newaxis, :]
                q3 = (k3/knl[np.newaxis, :])[np.newaxis, np.newaxis, :, :]            

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

        if not hasattr(self, 'gamman'):
            gamman = 10**(0.182+0.57*neff)
            an = 10**(-2.167-2.944*logsigma8z-1.106*logsigma8z**2-2.865*logsigma8z**3-0.310*r1**gamman)*self.cosmo['h']**2
            bn = 10**(-3.428 - 2.681*logsigma8z + 1.624*logsigma8z**2 - 0.095*logsigma8z**3)*self.cosmo['h']**2
            cn = 10**(0.159 - 1.107*neff)
            alphan = np.minimum(10**(-4.348 - 3.006*neff - 0.5745*neff**2 + 10**(-0.9+0.2*neff)*r2**2), 1 - 2./3*self.cosmo['n_s'])
            betan = 10**(-1.731-2.845*neff-1.4995*neff**2-0.2811*neff**3+0.007*r2)

            B1h = (1./(an*q1**alphan + bn*q1**betan)) * (1./(1. + (cn*q1)**-1)) * \
                    (1./(an*q2**alphan + bn*q2**betan)) * (1./(1. + (cn*q2)**-1)) * \
                    (1./(an*q3**alphan + bn*q3**betan)) *  (1./(1. + (cn*q3)**-1)) 
        else:
            an = 10**(-2.167-2.944*self.logsigma8z-1.106*self.logsigma8z**2-2.865*self.logsigma8z**3-0.310*r1**self.gamman)*self.cosmo['h']**2
            alphan = np.minimum(10**(-4.348 - 3.006*self.neff_arr - 0.5745*self.neff_arr**2 + 10**(-0.9+0.2*self.neff_arr)*r2**2), 1 - 2./3*self.cosmo['n_s'])
            betan = 10**(-1.731-2.845*self.neff_arr-1.4995*self.neff_arr**2-0.2811*self.neff_arr**3+0.007*r2)

            B1h = (1./(an*q1**alphan + self.bn*q1**betan)) * (1./(1. + (self.cn*q1)**-1)) * \
                    (1./(an*q2**alphan + self.bn*q2**betan)) * (1./(1. + (self.cn*q2)**-1)) * \
                    (1./(an*q3**alphan + self.bn*q3**betan)) *  (1./(1. + (self.cn*q3)**-1))   

        return B1h

    def I(self, q, en):
        '''
        Compute the I function.
        See: https://arxiv.org/abs/1911.07886.
        params:
        q: array, wavenumbers
        en: array, bihalofit en parameter
        '''

        I = 1./(1. + en*q)

        return I

    def PE(self, q, Pklin, fn, gn, hn, mn, mun, nn, nun, pn):
        '''
        Compute the enhanced power spectrum.
        See: https://arxiv.org/abs/1911.07886.
        params:
        q: array, wavenumbers
        Pklin: array, linear power spectrum
        fn, gn, hn, mn, mun, nn, nun, pn: array, bihalofit parameters
        '''

        PE = (1.+fn*q**2)/(1.+gn*q+hn*q**2)*Pklin + 1./(mn*q**mun+nn*q**nun)*1./(1.+(pn*q)**-3)

        return PE

    def B3h(self, k1, k2, k3, a, Pk2d, knl, neff, logsigma8z, Omegamz, cache=False):
        '''
        Compute the 3-halo term of the bispectrum using the bihalofit fitting function.
        See: https://arxiv.org/abs/1911.07886.
        params:
        k1, k2, k3: arrays, wavenumbers
        a: array, scale factor
        Pk2d: CCL Pk2d power spectrum object
        knl: array, non-linear scale
        neff: array, effective spectral index
        logsigma8z: array, log10 of sigma8 at redshift z
        Omegamz: array, Omega_matter at redshift z
        '''

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            q1 = (k1/knl[np.newaxis, :])[:, np.newaxis, np.newaxis, :] 
            q2 = (k2/knl[np.newaxis, :])[np.newaxis, :, np.newaxis, :]
            q3 = (k3/knl[np.newaxis, :])[np.newaxis, np.newaxis, :, :]
        else:
            q1 = k1/knl
            q2 = k2/knl 
            q3 = k3/knl

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            Pklin1 = np.diag(Pk2d(k=k1, a=a))[np.newaxis, np.newaxis, np.newaxis, :] 
            if not cache:
                print('Cache is False')
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

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            k1 = k1[np.newaxis, np.newaxis, np.newaxis, :]
            k2 = k2[np.newaxis, :, np.newaxis, :]
            k3 = k3[np.newaxis, np.newaxis, :, :]

        if not hasattr(self, 'fn'):
            fn = 10**(-10.533-16.838*neff-9.3048*neff**2-1.8263*neff**3)
            gn = 10**(2.787 + 2.405*neff+0.4577*neff**2)
            hn = 10**(-1.118-0.394*neff)
            mn = 10**(-2.605-2.434*logsigma8z+5.71*logsigma8z**2)*self.cosmo['h']**3       
            nn = 10**(-4.468-3.08*logsigma8z+1.035*logsigma8z**2)*self.cosmo['h']**3
            mun = 10**(15.312+22.977*neff+10.9579*neff**2+1.6586*neff**3)
            nun = 10**(1.347+1.246*neff+0.4525*neff**2)
            pn = 10**(0.071-0.433*neff)
            dn = 10**(-0.483+0.892*logsigma8z-0.086*Omegamz)
            en = 10**(-0.632+0.646*neff)

            Ik1 = self.I(q1, en)
            Ik2 = self.I(q2, en)
            Ik3 = self.I(q3, en)

            PE1 = self.PE(q1, Pklin1, fn, gn, hn, mn, mun, nn, nun, pn)
            PE2 = self.PE(q2, Pklin2, fn, gn, hn, mn, mun, nn, nun, pn)
            PE3 = self.PE(q3, Pklin3, fn, gn, hn, mn, mun, nn, nun, pn)

            B3h = 2*((bs.F2(k1, k2, k3) + dn*q3)*Ik1*Ik2*Ik3*PE1*PE2 + \
                    (bs.F2(k2, k3, k1) + dn*q1)*Ik2*Ik3*Ik1*PE2*PE3 + \
                    (bs.F2(k3, k1, k2) + dn*q2)*Ik3*Ik1*Ik2*PE3*PE1)

        else:
            Ik1 = self.I(q1, self.en)
            if cache:
                if not hasattr(self, 'Ik2'):
                    self.Ik2 = self.I(q2, self.en)
                    self.Ik3 = self.I(q3, self.en)
                    Ik2 = self.Ik2
                    Ik3 = self.Ik3
                else:
                    Ik2 = self.Ik2
                    Ik3 = self.Ik3
            else:
                Ik2 = self.I(q2, self.en)
                Ik3 = self.I(q3, self.en)

            PE1 = self.PE(q1, Pklin1, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, 
                          self.nun, self.pn)
            if cache:
                if not hasattr(self, 'PE2'):
                    Pklin2 = np.array([np.diag(Pk2d(k=np.squeeze(k2)[i, :], a=a)) for i in range(np.squeeze(k2).shape[0])])[np.newaxis, :, np.newaxis, :]
                    Pklin3 = np.array([np.diag(Pk2d(k=np.squeeze(k3)[i, :], a=a)) for i in range(np.squeeze(k3).shape[0])])[np.newaxis, np.newaxis, :, :]
                    self.PE2 = self.PE(q2, Pklin2, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, 
                                      self.nun, self.pn)
                    self.PE3 = self.PE(q3, Pklin3, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, 
                                      self.nun, self.pn)
                    PE2 = self.PE2
                    PE3 = self.PE3
                else:
                    PE2 = self.PE2
                    PE3 = self.PE3
            else:
                PE2 = self.PE(q2, Pklin2, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, self.nun, self.pn)
                PE3 = self.PE(q3, Pklin3, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, self.nun, self.pn)

            B3h = 2*((bs.F2(k1, k2, k3) + self.dn*q3)*Ik1*Ik2*Ik3*PE1*PE2 + \
                    (bs.F2(k2, k3, k1) + self.dn*q1)*Ik2*Ik3*Ik1*PE2*PE3 + \
                    (bs.F2(k3, k1, k2) + self.dn*q2)*Ik3*Ik1*Ik2*PE3*PE1)

        return B3h

    def Bk(self, k1, k2, k3, a, Pk2d=None, cache=False):
        ''' 
        Compute the bispectrum using the bihalofit fitting function.
        See: https://arxiv.org/abs/1911.07886.
        params:
        k1, k2, k3: arrays, wavenumbers
        a: array, scale factor
        Pk2d: CCL Pk2d power spectrum object
        '''

        if Pk2d is None:
            Pk2d = bs.init_Pk2d(self.cosmo)

        if not hasattr(self, 'logsigma8z'):
            logsigma8z = np.log10(np.array([ccl.power.sigmaR(self.cosmo, 8./self.cosmo['h'], a=ai, p_of_k_a=Pk2d) for ai in a]))
            Omegamz = ccl.omega_x(self.cosmo, a, 'matter')

            knl = self.knl(a)
            neff = self.neff(a)

            B1 = self.B1h(k1, k2, k3, a, knl, neff, logsigma8z)  

            B3 = self.B3h(k1, k2, k3, a, Pk2d, knl, neff, logsigma8z, Omegamz)

        else:
            B1 = self.B1h(k1, k2, k3, a, self.knl_arr, self.neff_arr, self.logsigma8z, cache)  

            B3 = self.B3h(k1, k2, k3, a, Pk2d, self.knl_arr, self.neff_arr, self.logsigma8z, self.Omegamz, cache)

        Bihft = B1 + B3

        return Bihft

    def Bl(self, k1, k2, k3, a, tr_weights, Pk2d=None):

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            q1 = (k1/self.knl_arr[np.newaxis, :])[:, np.newaxis, np.newaxis, :] 
            q2 = (k2/self.knl_arr[np.newaxis, :])[np.newaxis, :, np.newaxis, :]
            q3 = (k3/self.knl_arr[np.newaxis, :])[np.newaxis, np.newaxis, :, :]

        else:
            q1 = k1/self.knl_arr
            q2 = k2/self.knl_arr 
            q3 = k3/self.knl_arr

        if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
            Pklin1 = np.array([np.diag(Pk2d(k=k1[i, :], a=a)) for i in range(k2.shape[0])])[np.newaxis, :, np.newaxis, :]
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

        # if k1.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
        #     k1 = k1[:, np.newaxis, np.newaxis, :]
        #     k2 = k2[np.newaxis, :, np.newaxis, :]
        #     k3 = k3[np.newaxis, np.newaxis, :, :]

        Ik1 = self.I(q1, self.en)
        Ik2 = self.I(q2, self.en)
        Ik3 = self.I(q3, self.en)

        PE1 = self.PE(q1, Pklin1, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, self.nun, self.pn)
        PE2 = self.PE(q2, Pklin2, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, self.nun, self.pn)
        PE3 = self.PE(q3, Pklin3, self.fn, self.gn, self.hn, self.mn, self.mun, self.nn, self.nun, self.pn)

        for i in range(k1.shape[0]):
            k1_curr = k1[i, :]

            if k1_curr.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
                k1_mesh = np.meshgrid(np.array((k1_curr*self.chi_arr)[0]), (k2*self.chi_arr)[:, 0], 
                                (k3*self.chi_arr)[:, 0], indexing='ij')
                ksorted = np.sort(np.stack((k1_mesh[0][0, :, :], k1_mesh[1][0, :, :], 
                            k1_mesh[2][0, :, :])), axis=0)
                kmin = ksorted[0]
                kmid = ksorted[1]
                kmax = ksorted[-1]
                r1 = (kmin/kmax)[np.newaxis, :, :, np.newaxis]
                r2 = ((kmid+kmin-kmax)/kmax)[np.newaxis, :, :, np.newaxis]

            else:

                ksorted = np.sort([k1_curr, k2, k3], axis=0)
                kmin = ksorted[0]
                kmid = ksorted[1]
                kmax = ksorted[-1]
                r1 = kmin/kmax
                r2 = (kmid+kmin-kmax)/kmax

            an = 10**(-2.167-2.944*self.logsigma8z-1.106*self.logsigma8z**2-2.865*self.logsigma8z**3-0.310*r1**self.gamman)*self.cosmo['h']**2
            alphan = np.minimum(10**(-4.348 - 3.006*self.neff_arr - 0.5745*self.neff_arr**2 + 10**(-0.9+0.2*self.neff_arr)*r2**2), 1 - 2./3*self.cosmo['n_s'])
            betan = 10**(-1.731-2.845*self.neff_arr-1.4995*self.neff_arr**2-0.2811*self.neff_arr**3+0.007*r2)

            B1h = (1./(an*q1[i, :]**alphan + self.bn*q1**betan)) * (1./(1. + (self.cn*q1[i, :])**-1)) * \
                    (1./(an*q2**alphan + self.bn*q2**betan)) * (1./(1. + (self.cn*q2)**-1)) * \
                    (1./(an*q3**alphan + self.bn*q3**betan)) *  (1./(1. + (self.cn*q3)**-1)) 

            if k1_curr.ndim == 2 or k2.ndim == 2 or k3.ndim == 2:
                k1 = k1[np.newaxis, np.newaxis, np.newaxis, :]
                k2 = k2[np.newaxis, :, np.newaxis, :]
                k3 = k3[np.newaxis, np.newaxis, :, :]

            B3h = 2*((bs.F2(k1_curr, k2, k3) + self.dn*q3)*Ik1[i, :]*Ik2*Ik3*PE1[i, :]*PE2 + \
                (bs.F2(k2, k3, k1_curr) + self.dn*q1)*Ik2*Ik3*Ik1[i, :]*PE2*PE3 + \
                (bs.F2(k3, k1_curr, k2) + self.dn*q2)*Ik3*Ik1[i, :]*Ik2*PE3*PE1[i, :])

            Bk_intg = B1h + B3h

            integ = tr_weights*Bk_intg   
            Bl[i, :, :] = np.trapz(integ, self.chi_arr, axis=-1)

        return Bl
