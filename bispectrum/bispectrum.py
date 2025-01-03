import numpy as np 
import pyccl as ccl
from functools import partial
import logging
import sys
import bihalofit as bhf

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def init_Pk2d(cosmo, a_arr=None, k_arr=None, nonlinear=True):
    '''
    Initialize the CCL Pk2d power spectrum object.
    params:
    cosmo: ccl.Cosmology object
    a_arr: array, scale factor
    k_arr: array, wavenumbers
    nonlinear: bool, if True, use the non-linear power spectrum
    '''

    extrap_order_lok=1
    extrap_order_hik=2

    # k sampling
    if k_arr is None:
        log10k_min=-4
        log10k_max=2
        nk_per_decade=20
        nk_total = int((log10k_max - log10k_min) * nk_per_decade)
        k_arr = np.logspace(log10k_min, log10k_max, nk_total)

    # a sampling
    if a_arr is None:
        a_arr = ccl.get_pk_spline_a()

    pk = np.array([ccl.linear_matter_power(cosmo, k_arr, a=ai) for ai in a_arr])

    # Build interpolator
    pk2d = ccl.Pk2D(a_arr=a_arr,
                lk_arr=np.log(k_arr),
                pk_arr=pk,
                is_logp=False,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik)

    return pk2d

def F2(k1, k2, k3):
    '''
    Compute the bispectrum kernel F2.
    params:
    k1, k2, k3: array, wavenumbers
    '''

    cos12 = (k3**2 - k1**2 - k2**2) / (2*k1*k2)

    F2 = 5./7 + 0.5*cos12*(k1/k2+k2/k1) + 2./7*cos12**2

    return F2

def Bmmm(k1, k2, k3, a, Pk2d):
    '''
    Compute the 3D matter bispectrum.
    params:
    k1, k2, k3: array, wavenumbers
    a: array, scale factor
    Pk2d: ccl.Pk2D object
    '''

    if len(a) == 1 and np.size(k1) > 1:
        Pk1 = Pk2d(k=k1, a=a)[0]
        Pk2 = Pk2d(k=k2, a=a)[0]
        Pk3 = Pk2d(k=k3, a=a)[0]
    else:
        Pk1 = np.diag(Pk2d(k=k1, a=a))
        Pk2 = np.diag(Pk2d(k=k2, a=a))
        Pk3 = np.diag(Pk2d(k=k3, a=a))        

    B = 2*F2(k1, k2, k3)*Pk1*Pk2 + \
        2*F2(k2, k3, k1)*Pk2*Pk3 + \
        2*F2(k3, k1, k2)*Pk1*Pk3

    return B

def Bmmm_ev_test(k1, k2, k3, a, Pk1, Pk2, Pk3):
    '''
    Compute the 3D matter bispectrum from precomputed quantities.
    params:
    k1, k2, k3: array, wavenumbers
    a: array, scale factor
    Pk1, Pk2, Pk3: array, power spectra
    '''

    B = 2*F2(k1, k2, k3)*Pk1*Pk2 + \
        2*F2(k2, k3, k1)*Pk2*Pk3 + \
        2*F2(k3, k1, k2)*Pk1*Pk3

    return B

# def Bmmm_ev(k1, k2, k3, a, Pk2d):

#     # Pk1 = np.array([np.diag(Pk2d(k=k1[i, :], a=a)) for i in range(k1.shape[0])])[:, np.newaxis, np.newaxis, :] 
#     Pk1 = np.diag(Pk2d(k=k1, a=a))[np.newaxis, np.newaxis, np.newaxis, :] 
#     Pk2 = np.array([np.diag(Pk2d(k=k2[i, :], a=a)) for i in range(k2.shape[0])])[np.newaxis, :, np.newaxis, :]
#     Pk3 = np.array([np.diag(Pk2d(k=k3[i, :], a=a)) for i in range(k3.shape[0])])[np.newaxis, np.newaxis, :, :]       

#     k1 = k1[np.newaxis, np.newaxis, np.newaxis, :]
#     k2 = k2[np.newaxis, :, np.newaxis, :]
#     k3 = k3[np.newaxis, np.newaxis, :, :]

#     B = 2*F2(k1, k2, k3)*Pk1*Pk2 + \
#         2*F2(k2, k3, k1)*Pk2*Pk3 + \
#         2*F2(k3, k1, k2)*Pk1*Pk3

#     return B

def Bmmg(cosmo, ptt1, ptt2, ptt3, k1, k2, k3, a, Pk2d, Bkm='tree'):
    '''
    Compute the 3D matter-matter-galaxy bispectrum.
    params:
    cosmo: ccl.Cosmology object
    ptt1, ptt2, ptt3: ccl.PTTracer objects
    k1, k2, k3: array, wavenumbers
    a: array, scale factor
    Pk2d: ccl.Pk2D object
    Bkm: str, bispectrum model, can be 'tree' or 'halofit'
    '''

    cos12 = (k3**2 - k1**2 - k2**2) / (2*k1*k2)
    zs = 1./a - 1.

    if len(a) == 1 and np.size(k1) > 1:
        Pk1 = Pk2d(k=k1, a=a)[0]
        Pk2 = Pk2d(k=k2, a=a)[0]
    else:
        Pk1 = np.diag(Pk2d(k=k1, a=a))
        Pk2 = np.diag(Pk2d(k=k2, a=a))  

    if Bkm == 'tree':
        Bmmm_temp = Bmmm(k1, k2, k3, a, Pk2d)
    else:
        Bmmm_temp = bhf.Bihalofit(cosmo, k1, k2, k3, a, Pk2d)

    B = ptt3.b1(zs)*Bmmm_temp + \
        +(ptt3.b2(zs) + 2*ptt3.bk2(zs)*(cos12**2 - 1./3))*Pk1*Pk2

    return B

def Bggm(cosmo, ptt1, ptt2, ptt3, k1, k2, k3, a, Pk2d, ndens=None, Bkm='tree'):
    '''
    Compute the 3D matter-galaxy-galaxy bispectrum.
    params:
    cosmo: ccl.Cosmology object
    ptt1: galaxy tracer 1 ccl.PTTracer object
    ptt2: galaxy tracer 2 ccl.PTTracer object
    ptt3: matter tracer ccl.PTTracer object
    k1, k2, k3: array, wavenumbers
    a: array, scale factor
    Pk2d: ccl.Pk2D object
    ndens: float, number density of the tracers, if not None, shot noise will be included
    Bkm: str, bispectrum model, can be 'tree' or 'halofit'
    '''

    cos12 = (k3**2 - k1**2 - k2**2) / (2*k1*k2)
    cos13 = (k2**2 - k1**2 - k3**2) / (2*k1*k3)

    if len(a) == 1 and np.size(k1) > 1:
        Pk1 = Pk2d(k=k1, a=a)[0]
        Pk2 = Pk2d(k=k2, a=a)[0]
        Pk3 = Pk2d(k=k3, a=a)[0]
    else:
        Pk1 = np.diag(Pk2d(k=k1, a=a))
        Pk2 = np.diag(Pk2d(k=k2, a=a))
        Pk3 = np.diag(Pk2d(k=k3, a=a))   

    zs = 1./a - 1.

    if Bkm == 'tree':
        Bmmm_temp = Bmmm(k1, k2, k3, a, Pk2d)
    else:
        Bmmm_temp = bhf.Bihalofit(cosmo, k1, k2, k3, a, Pk2d)

    B = ptt1.b1(zs)*ptt2.b1(zs)*Bmmm_temp + \
        + ptt1.b1(zs)*(ptt2.b2(zs) + 2*ptt2.bk2(zs)*(cos12**2 - 1./3))*Pk2*Pk3 + \
        + ptt2.b1(zs)*(ptt1.b2(zs) + 2*ptt1.bk2(zs)*(cos13**2 - 1./3))*Pk1*Pk3

    if ptt2 == ptt3:
        if ndens is not None:
            B += 2.*Pk1*ptt1.b1(zs)/ndens

    return B
   
def Bggg(cosmo, ptt1, ptt2, ptt3, k1, k2, k3, a, Pk2d, ndens=None, Bkm='tree'):
    '''
    Compute the 3D galaxy-galaxy-galaxy bispectrum.
    params:
    cosmo: ccl.Cosmology object
    ptt1, ptt2, ptt3: ccl.PTTracer objects
    k1, k2, k3: array, wavenumbers
    a: array, scale factor
    Pk2d: ccl.Pk2D object
    ndens: float, number density of the tracers, if not None, shot noise will be included
    Bkm: str, bispectrum model, can be 'tree' or 'halofit'
    '''

    cos12 = (k3**2 - k1**2 - k2**2) / (2*k1*k2)
    cos13 = (k2**2 - k1**2 - k3**2) / (2*k1*k3)
    cos23 = (k1**2 - k2**2 - k3**2) / (2*k2*k3)

    if len(a) == 1 and np.size(k1) > 1:
        Pk1 = Pk2d(k=k1, a=a)[0]
        Pk2 = Pk2d(k=k2, a=a)[0]
        Pk3 = Pk2d(k=k3, a=a)[0]
    else:
        Pk1 = np.diag(Pk2d(k=k1, a=a))
        Pk2 = np.diag(Pk2d(k=k2, a=a))
        Pk3 = np.diag(Pk2d(k=k3, a=a))   

    zs = 1./a - 1.

    if Bkm == 'tree':
        Bmmm_temp = Bmmm(k1, k2, k3, a, Pk2d)
    else:
        Bmmm_temp = bhf.Bihalofit(cosmo, k1, k2, k3, a, Pk2d)

    B = ptt1.b1(zs)*ptt2.b1(zs)*ptt3.b1(zs)*Bmmm_temp + \
        + ptt1.b1(zs)*ptt2.b1(zs)*(ptt3.b2(zs) + 2*ptt3.bk2(zs)*(cos12**2 - 1./3))*Pk1*Pk2 + \
        + ptt2.b1(zs)*ptt3.b1(zs)*(ptt1.b2(zs) + 2*ptt1.bk2(zs)*(cos23**2 - 1./3))*Pk2*Pk3 + \
        + ptt1.b1(zs)*ptt3.b1(zs)*(ptt2.b2(zs) + 2*ptt2.bk2(zs)*(cos13**2 - 1./3))*Pk1*Pk3 

    if ptt1 == ptt2 and ptt1 == ptt3:
        if ndens is not None:
            B += 1./ndens**2
            B += 2.*1./ndens*(ptt1.b1(zs) + ptt2.b1(zs) + ptt3.b1(zs))

    return B

def Bl(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm='tree', ndens=None):
    '''
    Compute the angular bispectrum in the Limber approximation.
    params:
    cosmo: ccl.Cosmology object
    tr1, tr2, tr3: ccl.Tracer objects
    ptt1, ptt2, ptt3: ccl.PTTracer objects
    l1, l2, l3: array, multipoles
    Bkm: str, bispectrum model, can be 'tree' or 'halofit'
    ndens: float, number density of the tracers, if not None, shot noise will be included
    '''

    assert l1.shape[0] == l2.shape[0] == l3.shape[0], 'Shape mismatch of ell arrays.'

    z_arr = np.linspace(0.001, 3, 300)
    a_arr = 1./(1+z_arr)
    chi_arr = ccl.comoving_radial_distance(cosmo, a_arr)

    Pk2d = init_Pk2d(cosmo)

    pg = []
    pm = []
    lg = []
    lm = []

    if ptt1.type == 'NC':
        pg.append(ptt1)
        lg.append(l1)
    else:
        pm.append(ptt1)
        lm.append(l1)
    if ptt2.type == 'NC':
        pg.append(ptt2)
        lg.append(l2)
    else:   
        pm.append(ptt2)
        lm.append(l2)
    if ptt3.type == 'NC':  
        pg.append(ptt3)
        lg.append(l3)
    else:   
        pm.append(ptt3)  
        lm.append(l3)

    bias_fac = 1

    if len(pg) == 0:

        logger.info('Computing Bmmm')
        if Bkm == 'tree':
            Bk_part = partial(Bmmm, a=a_arr, Pk2d=Pk2d)
        else:
            bihalofit = bhf.Bihalofit(cosmo, a=a_arr, Pk2d=Pk2d)
            Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lm[2]
    
    elif len(pg) == 1:

        logger.info('Computing Bmmg')  

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk_part = partial(Bmmm, a=a_arr, Pk2d=Pk2d)
            else:
                bihalofit = bhf.Bihalofit(cosmo)
                Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
            bias_fac = pg[0].b1(0)
        else:
            Bk_part = partial(Bmmg, cosmo=cosmo, ptt1=pm[0], ptt2=pm[1], ptt3=pg[0], a=a_arr, Pk2d=Pk2d, Bkm=Bkm)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lg[0]

    elif len(pg) == 2:

        logger.info('Computing Bggm')

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk_part = partial(Bmmm, a=a_arr, Pk2d=Pk2d)
            else:
                bihalofit = bhf.Bihalofit(cosmo)
                Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
            bias_fac = pg[0].b1(0)**2
        else:
            Bk_part = partial(Bggm, cosmo=cosmo, ptt1=pg[0], ptt2=pg[1], ptt3=pm[0], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lg[0]
        l2 = lg[1]
        l3 = lm[0]

    else:

        logger.info('Computing Bggg')

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk_part = partial(Bmmm, a=a_arr, Pk2d=Pk2d)
            else:
                bihalofit = bhf.Bihalofit(cosmo, a=a_arr, Pk2d=Pk2d)
                Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d)
            bias_fac = pg[0].b1(0)**3
        else:
            Bk_part = partial(Bggg, cosmo=cosmo, ptt1=pg[0], ptt2=pg[1], ptt3=pg[2], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lg[0]
        l2 = lg[1]
        l3 = lg[2]

    Bl = np.zeros_like(l1)
    pref = tr1.get_kernel(chi_arr)[0]*tr2.get_kernel(chi_arr)[0]*tr3.get_kernel(chi_arr)[0]/chi_arr**4
    for i in range(len(l1)):
        B = Bk_part(k1=l1[i]/chi_arr, k2=l2[i]/chi_arr, k3=l3[i]/chi_arr)

        integ = pref*B
        Bl[i] = np.trapz(integ, chi_arr)

    return bias_fac*Bl

def Bl_ev(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm='tree', ndens=None):
    '''
    Compute the angular bispectrum in the Limber approximation on a grid of ells. 
    This is a convenience function for the FSB calculation.
    params:
    cosmo: ccl.Cosmology object
    tr1, tr2, tr3: ccl.Tracer objects
    ptt1, ptt2, ptt3: ccl.PTTracer objects
    l1, l2, l3: array, multipoles
    Bkm: str, bispectrum model, can be 'tree' or 'halofit'
    ndens: float, number density of the tracers, if not None, shot noise will be included
    '''

    z_arr = np.linspace(0.001, 3, 300)
    a_arr = 1./(1+z_arr)
    chi_arr = ccl.comoving_radial_distance(cosmo, a_arr)

    Pk2d = init_Pk2d(cosmo)

    pg = []
    pm = []
    lg = []
    lm = []

    if ptt1.type == 'NC':
        pg.append(ptt1)
        lg.append(l1)
    else:
        pm.append(ptt1)
        lm.append(l1)
    if ptt2.type == 'NC':
        pg.append(ptt2)
        lg.append(l2)
    else:   
        pm.append(ptt2)
        lm.append(l2)
    if ptt3.type == 'NC':  
        pg.append(ptt3)
        lg.append(l3)
    else:   
        pm.append(ptt3)  
        lm.append(l3)

    bias_fac = 1

    if len(pg) == 0:

        logger.info('Computing Bmmm')
        if Bkm == 'tree':
            Bk_part = partial(Bmmm_ev_test, a=a_arr)
        else:
            bihalofit = bhf.Bihalofit(cosmo, a=a_arr, Pk2d=Pk2d)
            Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
            # Bl_part = partial(bihalofit.Bl, a=a_arr, Pk2d=Pk2d)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lm[2]
    
    elif len(pg) == 1:

        logger.info('Computing Bmmg')  

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk_part = partial(Bmmm_ev_test, a=a_arr)
            else:
                bihalofit = bhf.Bihalofit(cosmo, a=a_arr, Pk2d=Pk2d)
                Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
            bias_fac = pg[0].b1(0)
        else:
            Bk_part = partial(Bmmg, cosmo=cosmo, ptt1=pm[0], ptt2=pm[1], ptt3=pg[0], a=a_arr, Pk2d=Pk2d, Bkm=Bkm)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lg[0]

    elif len(pg) == 2:

        logger.info('Computing Bggm')

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk_part = partial(Bmmm_ev_test, a=a_arr)
            else:
                bihalofit = bhf.Bihalofit(cosmo, a=a_arr, Pk2d=Pk2d)
                Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
            bias_fac = pg[0].b1(0)**2
        else:
            Bk_part = partial(Bggm, cosmo=cosmo, ptt1=pg[0], ptt2=pg[1], ptt3=pm[0], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lg[0]
        l2 = lg[1]
        l3 = lm[0]

    else:

        logger.info('Computing Bggg')

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk_part = partial(Bmmm_ev_test, a=a_arr)
            else:
                bihalofit = bhf.Bihalofit(cosmo, a=a_arr, Pk2d=Pk2d)
                Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d, cache=True)
            bias_fac = pg[0].b1(0)**3
        else:
            Bk_part = partial(Bggg, cosmo=cosmo, ptt1=pg[0], ptt2=pg[1], ptt3=pg[2], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lg[0]
        l2 = lg[1]
        l3 = lg[2]

    Bl = np.zeros((l1.shape[0], l2.shape[0], l3.shape[0]))
    pref = tr1.get_kernel(chi_arr)[0]*tr2.get_kernel(chi_arr)[0]*tr3.get_kernel(chi_arr)[0]/chi_arr**4
    if Bkm == 'tree':
        k1 = l1[:, np.newaxis]/chi_arr
        k2 = l2[:, np.newaxis]/chi_arr
        k3 = l3[:, np.newaxis]/chi_arr
        Pk1 = np.array([np.diag(Pk2d(k=k1[i, :], a=a_arr)) for i in range(k1.shape[0])])
        Pk2 = np.array([np.diag(Pk2d(k=k2[i, :], a=a_arr)) for i in range(k2.shape[0])])[np.newaxis, :, np.newaxis, :]
        Pk3 = np.array([np.diag(Pk2d(k=k3[i, :], a=a_arr)) for i in range(k3.shape[0])])[np.newaxis, np.newaxis, :, :] 
        k2 = k2[np.newaxis, :, np.newaxis, :]
        k3 = k3[np.newaxis, np.newaxis, :, :]    

        Pk1 = Pk1[:, np.newaxis, np.newaxis, :]       
        k1 = k1[:, np.newaxis, np.newaxis, :]

        for i in range(len(l1)):
            Bk_intg = Bk_part(k1=k1[i, :], k2=k2, k3=k3, Pk1=Pk1[i, :], Pk2=Pk2, Pk3=Pk3)
            integ = pref*Bk_intg   
            Bl[i, :, :] = np.trapz(integ, chi_arr, axis=-1)
    else:
        k1 = l1[:, np.newaxis]/chi_arr
        k2 = l2[:, np.newaxis]/chi_arr
        k3 = l3[:, np.newaxis]/chi_arr

        # Bl = Bl_part(k1=k1, k2=k2, k3=k3, tr_weights=pref)
        for i in range(len(l1)):
            Bk_intg = Bk_part(k1=k1[i], k2=k2, k3=k3)

            integ = pref*Bk_intg   
            Bl[i, :, :] = np.trapz(integ, chi_arr, axis=-1)

    return bias_fac*Bl

def Bk(cosmo, ptt1, ptt2, ptt3, a, k1, k2, k3, Pk2d=None, Bkm='tree', ndens=None):
    '''
    Compute the 3D  bispectrum.
    params:
    cosmo: CCL Cosmology object
    ptt1, ptt2, ptt3: ccl.Tracer objects
    a: array, scale factor
    k1, k2, k3: array, wavenumbers
    Pk2d: ccl.Pk2D object
    Bkm: str, bispectrum model, can be 'tree' or 'halofit'
    ndens: float, number density of the tracers, if not None, shot noise will be included
    '''

    if Pk2d is None:
        Pk2d = init_Pk2d(cosmo)

    pg = []
    pm = []
    kg = []
    km = []

    if ptt1.type == 'NC':
        pg.append(ptt1)
        kg.append(k1)
    else:
        pm.append(ptt1)
        km.append(k1)
    if ptt2.type == 'NC':
        pg.append(ptt2)
        kg.append(k2)
    else:   
        pm.append(ptt2)
        km.append(k2)
    if ptt3.type == 'NC':  
        pg.append(ptt3)
        kg.append(k3)
    else:   
        pm.append(ptt3)  
        km.append(k3)

    bias_fac = 1

    if len(pg) == 0:

        logger.info('Computing Bmmm')
        if Bkm == 'tree':
            Bk = Bmmm(km[0], km[1], km[2], a, Pk2d)
        else:
            Bhf = bhf.Bihalofit(cosmo, a, Pk2d)
            Bk = Bhf.Bk(km[0], km[1], km[2], a, Pk2d)
    
    elif len(pg) == 1:

        logger.info('Computing Bmmg')  

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk = Bmmm(km[0], km[1], kg[0], a, Pk2d)
            else:
                Bhf = bhf.Bihalofit(cosmo, a, Pk2d)
                Bk = Bhf.Bk(km[0], km[1], kg[0], a, Pk2d)
            bias_fac = pg[0].b1(0)
        else:
            Bk = Bmmg(cosmo, pm[0], pm[1], pg[0], km[0], km[1], kg[0], a, Pk2d, Bkm)

    elif len(pg) == 2:

        logger.info('Computing Bggm')

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk = Bmmm(kg[0], kg[1], km[0], a, Pk2d)
            else:
                Bhf = bhf.Bihalofit(cosmo, a, Pk2d)
                Bk = Bhf.Bk(kg[0], kg[1], km[0], a, Pk2d)
            bias_fac = pg[0].b1(0)**2
        else:
            Bk = Bggm(cosmo, pg[0], pg[1], pm[0], kg[0], kg[1], km[0], a, Pk2d, ndens, Bkm)

    else:

        logger.info('Computing Bggg')

        if pg[0].b2(0) == 0.:
            logger.info('Assuming linear bias for the galaxy tracer.')
            if Bkm == 'tree':
                Bk = Bmmm(km[0], km[1], km[2], a, Pk2d)
            else:
                Bhf = bhf.Bihalofit(cosmo, a, Pk2d)
                Bk = Bhf.Bk(km[0], km[1], km[2], a, Pk2d)
            bias_fac = pg[0].b1(0)**3
        else:
            Bk = Bggg(cosmo, pg[0], pg[1], pg[2], kg[0], kg[1], kg[2], a, Pk2d, ndens, Bkm)
   
    return bias_fac*Bk