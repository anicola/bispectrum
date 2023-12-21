import numpy as np 
import pyccl as ccl
from functools import partial
import logging
import sys
import bihalofit as bhf

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def init_Pk2d(cosmo, a_arr=None, k_arr=None, nonlinear=True):

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

    cos12 = (k3**2 - k1**2 - k2**2) / (2*k1*k2)

    F2 = 5./7 + 0.5*cos12*(k1/k2+k2/k1) + 2./7*cos12**2

    return F2

def Bmmm(k1, k2, k3, a, Pk2d):

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

def Bmmm_ev(k1, k2, k3, a, Pk2d):

    Pk1 = np.array([np.diag(Pk2d(k=k1[i, :], a=a)) for i in range(k1.shape[0])])[:, np.newaxis, np.newaxis, :] 
    Pk2 = np.array([np.diag(Pk2d(k=k2[i, :], a=a)) for i in range(k2.shape[0])])[np.newaxis, :, np.newaxis, :]
    Pk3 = np.array([np.diag(Pk2d(k=k3[i, :], a=a)) for i in range(k3.shape[0])])[np.newaxis, np.newaxis, :, :]       

    k1 = k1[:, np.newaxis, np.newaxis, :]
    k2 = k2[np.newaxis, :, np.newaxis, :]
    k3 = k3[np.newaxis, np.newaxis, :, :]

    B = 2*F2(k1, k2, k3)*Pk1*Pk2 + \
        2*F2(k2, k3, k1)*Pk2*Pk3 + \
        2*F2(k3, k1, k2)*Pk1*Pk3

    return B

def Bmmg(cosmo, ptt1, ptt2, ptt3, k1, k2, k3, a, Pk2d, Bkm='tree'):

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

def Bmgg(cosmo, ptt1, ptt2, ptt3, k1, k2, k3, a, Pk2d, ndens=None, Bkm='tree'):

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

    B = ptt2.b1(zs)*ptt3.b1(zs)*Bmmm_temp + \
        + ptt2.b1(zs)*(ptt3.b2(zs) + 2*ptt3.bk2(zs)*(cos12**2 - 1./3))*Pk1*Pk2 + \
        + ptt3.b1(zs)*(ptt2.b2(zs) + 2*ptt2.bk2(zs)*(cos13**2 - 1./3))*Pk1*Pk3

    if ptt2 == ptt3:
        if ndens is not None:
            B += 2.*Pk1*ptt2.b1(zs)/ndens

    return B
   
def Bggg(cosmo, ptt1, ptt2, ptt3, k1, k2, k3, a, Pk2d, ndens=None, Bkm='tree'):

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

    if len(pg) == 0:

        logger.info('Computing Bmmm')
        if Bkm == 'tree':
            Bk_part = partial(Bmmm, a=a_arr, Pk2d=Pk2d)
        else:
            bihalofit = bhf.Bihalofit(cosmo)
            Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lm[2]
    
    elif len(pg) == 1:

        logger.info('Computing Bmmg')  
        Bk_part = partial(Bmmg, cosmo=cosmo, ptt1=pm[0], ptt2=pm[1], ptt3=pg[0], a=a_arr, Pk2d=Pk2d, Bkm=Bkm)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lg[0]

    elif len(pg) == 2:

        logger.info('Computing Bmgg')
        Bk_part = partial(Bmgg, cosmo=cosmo, ptt1=pm[0], ptt2=pg[0], ptt3=pg[1], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lm[0]
        l2 = lg[0]
        l3 = lg[1]

    else:

        logger.info('Computing Bggg')
        Bk_part = partial(Bggg, cosmo=cosmo, ptt1=pg[0], ptt2=pg[1], ptt3=pg[2], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lg[0]
        l2 = lg[1]
        l3 = lg[2]

    Bl = np.zeros_like(l1)
    for i in range(len(l1)):
        B = Bk_part(k1=l1[i]/chi_arr, k2=l2[i]/chi_arr, k3=l3[i]/chi_arr)

        integ = tr1.get_kernel(chi_arr)[0]*tr2.get_kernel(chi_arr)[0]*tr3.get_kernel(chi_arr)[0]/chi_arr**4*B
        Bl[i] = np.trapz(integ, chi_arr)

    return Bl

def Bl_ev(cosmo, tr1, tr2, tr3, ptt1, ptt2, ptt3, l1, l2, l3, Bkm='tree', ndens=None):

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

    if len(pg) == 0:

        logger.info('Computing Bmmm')
        if Bkm == 'tree':
            Bk_part = partial(Bmmm_ev, a=a_arr, Pk2d=Pk2d)
        else:
            bihalofit = bhf.Bihalofit(cosmo)
            Bk_part = partial(bihalofit.Bk, a=a_arr, Pk2d=Pk2d)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lm[2]
    
    elif len(pg) == 1:

        logger.info('Computing Bmmg')  
        Bk_part = partial(Bmmg, cosmo=cosmo, ptt1=pm[0], ptt2=pm[1], ptt3=pg[0], a=a_arr, Pk2d=Pk2d, Bkm=Bkm)
        l1 = lm[0]
        l2 = lm[1]
        l3 = lg[0]

    elif len(pg) == 2:

        logger.info('Computing Bmgg')
        Bk_part = partial(Bmgg, cosmo=cosmo, ptt1=pm[0], ptt2=pg[0], ptt3=pg[1], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lm[0]
        l2 = lg[0]
        l3 = lg[1]

    else:

        logger.info('Computing Bggg')
        Bk_part = partial(Bggg, cosmo=cosmo, ptt1=pg[0], ptt2=pg[1], ptt3=pg[2], a=a_arr, Pk2d=Pk2d, Bkm=Bkm, ndens=ndens)
        l1 = lg[0]
        l2 = lg[1]
        l3 = lg[2]

    if Bkm != 'halofit':
        B = Bk_part(k1=l1[:, np.newaxis]/chi_arr, k2=l2[:, np.newaxis]/chi_arr, 
                k3=l3[:, np.newaxis]/chi_arr)

        integ = tr1.get_kernel(chi_arr)[0]*tr2.get_kernel(chi_arr)[0]*\
                tr3.get_kernel(chi_arr)[0]/chi_arr**4*B

        Bl = np.trapz(integ, chi_arr, axis=-1)

    else:
        Bk_intg = np.zeros((l1.shape[0], l2.shape[0], l3.shape[0], chi_arr.shape[0]))
        for i in range(len(l1)):
            Bk_intg[i, :, :, :] = Bk_part(k1=l1[i]/chi_arr, k2=l2[:, np.newaxis]/chi_arr,
                 k3=l3[:, np.newaxis]/chi_arr)

        integ = tr1.get_kernel(chi_arr)[0]*tr2.get_kernel(chi_arr)[0]*tr3.get_kernel(chi_arr)[0]/chi_arr**4*Bk_intg   

        Bl = np.trapz(integ, chi_arr, axis=-1)

    return Bl

def Bk(cosmo, ptt1, ptt2, ptt3, a, k1, k2, k3, Pk2d=None, Bkm='tree', ndens=None):

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

    if len(pg) == 0:

        logger.info('Computing Bmmm')
        if Bkm == 'tree':
            Bk = Bmmm(km[0], km[1], km[2], a, Pk2d)
        else:
            Bk = bhf.Bihalofit(cosmo, km[0], km[1], km[2], a, Pk2d)
    
    elif len(pg) == 1:

        logger.info('Computing Bmmg')
        Bk = Bmmg(cosmo, pm[0], pm[1], pg[0], km[0], km[1], kg[0], a, Pk2d, Bkm)

    elif len(pg) == 2:

        logger.info('Computing Bmgg')
        Bk = Bmgg(cosmo, pm[0], pg[0], pg[1], km[0], kg[0], kg[1], a, Pk2d, ndens, Bkm)

    else:

        logger.info('Computing Bggg')
        Bk = Bggg(cosmo, pg[0], pg[1], pg[2], kg[0], kg[1], kg[2], a, Pk2d, ndens, Bkm)
   
    return Bk