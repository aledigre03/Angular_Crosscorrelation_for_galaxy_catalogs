import numpy as np
import healpy
import pandas as pd
import Corrfunc

def read_target_catalogB(filename):
    """ """
    #high redshift catalog
    print(f"reading high redshift sample {filename}")
    catB = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    raB = catB['ra_gal']
    decB = catB['dec_gal']
    # cut the octant (a few galaxies are outside)
    selB1 = (raB > 0)&(raB < 5)&(decB > 0)&(decB < 5)
    catB = catB[selB1]
    #cut out very bright galaxies
    highredshift= catB['observed_redshift_gal']
    selB = highredshift > 0.9
    catB = catB[selB]
    return catB

def read_target_catalogA(filename):
    """ """
    #interloper sample
    print(f"reading Interloper sample {filename}")
    catA = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    raA = catA['ra_gal']
    decA = catA['dec_gal']
    # cut the octant (a few galaxies are outside)
    selA = (raA > 0)&(raA < 5)&(decA > 0)&(decA < 5)
    catA = catA[selA]
    #cut out very low redshift
    ref_mag_r_A = -2.5*np.log10(catA['lsst_r']) - 48.6
    sel = (ref_mag_r_A > 19)
    catA = catA[sel]
    return catA


def read_reference_catalog(filename):
    """ """
    print(f"reading reference sample {filename}")
    cat = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    ra = cat['ra_gal']
    dec = cat['dec_gal']
    # cut the octant (a few galaxies are outside)
    sel = (ra > 0)&(ra < 5)&(dec > 0)&(dec < 5)
    cat = cat[sel]
    #cut out very bright galaxies
    ref_mag_r = -2.5*np.log10(cat['lsst_r']) - 48.6
    sel = (ref_mag_r > 18)&(ref_mag_r < 19)
    cat = cat[sel]
    # randomize the catalog order
    cat = cat.sample(frac=1, ignore_index=True, random_state=3)
    return cat


def generate_random_catalog(size):
    """ """
    ra_random= np.random.uniform(0, 5, size)
    dec_random = np.random.uniform(0, 5, size)
    ran_cat = np.column_stack((ra_random, dec_random))

    return ran_cat


def generate_random_catalog(size):
    """Generating Random Catalog """
    ra_random= np.random.uniform(0, 5, size)
    dec_random = np.random.uniform(0, 5, size)
    ran_cat = np.column_stack((ra_random, dec_random))

    return ran_cat

def compute_wtheta(rat, dect, rar, decr, rand_ra, rand_dec, bins=None, weight_t=None, weight_r=None, weight_rand= None, nthreads=1):
    """Computing angular correlation function for a pixelized map """
    weight_type = None
    if weight_r is not None:
        weight_type = 'pair_product'
        print(f"data weights {weight_t.min()} {weight_t.max()}")
        print(f"data weights {weight_r.min()} {weight_r.max()}")
        print(f"rand weights {weight_rand.min()} {weight_rand.max()}")

    autocorr = 1
    crosscorr = 0

    # Cross pair counts in DtR
    DtR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                       RA1=rat, DEC1=dect, RA2=rand_ra, DEC2=rand_dec, weights1=weight_t, weights2=weight_rand,
                                       weight_type=weight_type)
    print("DtR cross correlation completed")

    # Cross pair counts in DrR
    DrR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                       RA1=rar, DEC1=decr, RA2=rand_ra, DEC2=rand_dec, weights1=weight_r, weights2=weight_rand,
                                       weight_type=weight_type)
    print("DrR cross correlation completed")

    # Cross pair counts in DtDr
    DtDr = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                        RA1=rat, DEC1=dect, RA2=rar, DEC2=decr, weights1=weight_t, weights2=weight_r,
                                        weight_type=weight_type)
    print("DtDr cross correlation completed")

    # Auto pairs counts in DtDt
    DtDt = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads, RA1=rat, DEC1=dect,
                                        binfile=bins, weights1=weight_t, weight_type=weight_type)
    print("DtDt auto correlation completed")

    # Auto pair counts in RR
    RR = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads, RA1=rand_ra, DEC1=rand_dec,
                                      binfile=bins, weights1=weight_rand, weight_type=weight_type)
    print("RR auto correlation completed")

    if weight_type == 'pair_product':
        print("modified pairs")
        DtDt_ = DtDt['npairs'] * DtDt['weightavg']
        DtDr_ = DtDr['npairs'] * DtDr['weightavg']
        DrR_ = DrR['npairs'] * DrR['weightavg']
        DtR_ = DtR['npairs'] * DtR['weightavg']
        RR_ = RR['npairs'] * RR['weightavg']
        ndata_target = np.sum(weight_t)
        ndata_reference= np.sum(weight_r)
        nrand = np.sum(weight_rand)
    else:
        DtDt_ = DtDt
        DrR_ = DrR
        DtDr_ = DtDr
        DtR_ = DtR
        RR_ = RR
        ndata_target = len(rat)
        ndata_reference = len(rar)
        nrand = len(rand_ra)

    # All the pair counts are done, get the angular correlation function
    wtheta_cross = Corrfunc.utils.convert_3d_counts_to_cf(ndata_target, ndata_reference, nrand, nrand,
                                                          DtDr_, DtR_, DrR_, RR_)

    wtheta_auto= Corrfunc.utils.convert_3d_counts_to_cf(ndata_target, ndata_target, nrand, nrand,
                                                        DtDt_, DtR_, DtR_, RR_)

    theta_min = DtDt['thetamin']
    theta_max = DtDt['thetamax']
    theta = (theta_min + theta_max)/2.
    return theta, wtheta_cross, wtheta_auto

def bin_count(indices):
    """"""
    counter = {}
    for i in indices:
        if i not in counter:
            counter[i] = 0
            counter[i] += 1
    pix_list = np.zeros(len(counter), dtype=int)
    count = np.zeros(len(counter), dtype=float)
    for i, p in enumerate(counter.keys()):
        pix_list[i] = p
        count[i] = counter[p]
    return pix_list, count

def pixelize(ra, dec, nside):
    """Compute pixelized map for array of points"""

    print(f"pixelizing map {len(ra)} {nside=}")
    pix = healpy.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    pix, count = bin_count(pix)
    ra_pix, dec_pix = healpy.pix2ang(nside, pix, nest=True, lonlat=True)
    return ra_pix, dec_pix, count

def compute_wtheta_healpix(rat, dect, rar, decr, rand_ra, rand_dec, bins, nthreads=1, nside=2048):
    """Compute angular correlation function w(theta) using the pixelized map method"""
    ra_pix_t, dec_pix_t, count_t = pixelize(rat, dect, nside=nside)
    ra_pix_r, dec_pix_r, count_r = pixelize(rar, decr, nside=nside)
    ra_pix_rand, dec_pix_rand, count_rand = pixelize(rand_ra, rand_dec, nside=nside)
    theta, wtheta_cross, wtheta_auto = compute_wtheta(ra_pix_t, dec_pix_t, ra_pix_r, dec_pix_r,
                                                      ra_pix_rand, dec_pix_rand, weight_t=count_t,
                                                      weight_r=count_r, weight_rand= count_rand,
                                                      bins=bins, nthreads=nthreads)

    return theta, wtheta_cross, wtheta_auto

#setup the bins (9)
bins = np.linspace(0.01, 1, 10)

#reference with known redshift: r<19,
#pool for interlopers: r>19
#17419 low redshift
#17418 Euclid Alpha

low_redshift_file = "your_catalog_name"
Euclid_Halpha_file = "your_catalog_name"
targ_cat1 = read_target_catalogB(Euclid_Halpha_file)     #B
targ_cat2 = read_target_catalogA(low_redshift_file)      #A2
ref_cat = read_reference_catalog(low_redshift_file)      #A1

#building catalog with topN% interlopers
topN= 0.1
print("selecting interlopers fraction. Chosen is {:.2f}".format(topN))
rabeta= targ_cat1['ra_gal']
Nbeta= len(rabeta)
Nalpha= int(((topN)/(1-topN))*Nbeta)
#select randomly Nalpha elements from catalogue
targ_cat2_alpha = targ_cat2.sample(n=Nalpha)
#sum two catalogues
targ_cat= pd.concat([targ_cat2_alpha, targ_cat1])
#randomize the final catalogue to avoid bias
targ_cat = targ_cat.sample(frac=1, ignore_index=True, random_state=3)


#correlation calculation
print("starting correlation calculation")
rat= targ_cat['ra_gal']
dect= targ_cat['dec_gal']
Nt= len(rat)
print("length target catalog is %d" % Nt)

rar=ref_cat['ra_gal']
decr= ref_cat['dec_gal']
Nr= len(rar)
print("length reference catalog is %d" % Nr)

#Random catalogue generation
rand_cat= generate_random_catalog(10*len(rat))
rand_ra, rand_dec = rand_cat[:, 0], rand_cat[:, 1]
rand_N = len(rand_ra)
print("random catalog generated, length is %d" % rand_N)


theta, wtheta_cross, wtheta_auto= compute_wtheta_healpix(rat, dect, rar, decr, rand_ra, rand_dec, bins)

print("wtheta_cross=", wtheta_cross)
print("wtheta_auto=", wtheta_auto)
print("theta=", theta)
