import numpy as np
import healpy
import Corrfunc
import pandas as pd
from matplotlib import pyplot as plt

def read_target_catalogB(filename):
    """ """
    #high redshift catalog
    print(f"reading high redshift sample {filename}")
    catB = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    raB = catB['ra_gal']
    decB = catB['dec_gal']
    # cut the octant (a few galaxies are outside)
    selB1 = (raB > 0)&(raB < 90)&(decB > 0)&(decB < 90)
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
    selA = (raA > 0)&(raA < 90)&(decA > 0)&(decA < 90)
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
    sel = (ra > 0)&(ra < 90)&(dec > 0)&(dec < 90)
    cat = cat[sel]
    #cut out very bright galaxies
    ref_mag_r = -2.5*np.log10(cat['lsst_r']) - 48.6
    sel = (ref_mag_r > 18)&(ref_mag_r < 19)
    cat = cat[sel]
    # randomize the catalog order
    cat = cat.sample(frac=1, ignore_index=True, random_state=3)
    return cat


def compute_wtheta(ra_targ, dec_targ, ra_ref, dec_ref, ra_int, dec_int, rand_ra, rand_dec, bins=None, weight_targ=None, weight_ref=None, weight_int=None, weight_rand=None, nthreads=1):
    """Return angular correlation function w(theta)"""
    weight_type = None
    if weight_targ is not None:
        weight_type = 'pair_product'

    autocorr = 1
    crosscorr = 0

    print("computing autocorrelations")

    DtDt = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads,
                                      RA1=ra_targ, DEC1=dec_targ,
                                      binfile=bins,
                                      weights1=weight_targ,
                                      weight_type=weight_type)

    DrDr = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads,
                                      RA1=ra_ref, DEC1=dec_ref,
                                      binfile=bins,
                                      weights1=weight_ref,
                                      weight_type=weight_type)

    DiDi = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads,
                                      RA1=ra_int, DEC1=dec_int,
                                      binfile=bins,
                                      weights1=weight_int,
                                      weight_type=weight_type)

    RR = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads,
                                      RA1=rand_ra, DEC1=rand_dec,
                                      binfile=bins,
                                      weights1=weight_rand,
                                      weight_type=weight_type)

    print("computing crosscorrelation")

    DtDr = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads,
                                      RA1=ra_targ, DEC1=dec_targ,
                                      RA2=ra_ref, DEC2=dec_ref,
                                      binfile=bins,
                                      weights1=weight_targ,
                                      weights2=weight_ref,
                                      weight_type=weight_type)

    print("computing random correlations")

    DtR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads,
                                      RA1=ra_targ, DEC1=dec_targ,
                                      RA2=rand_ra, DEC2=rand_dec,
                                      binfile=bins,
                                      weights1=weight_targ,
                                      weights2=weight_rand,
                                      weight_type=weight_type)

    DrR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads,
                                      RA1=ra_ref, DEC1=dec_ref,
                                      RA2=rand_ra, DEC2=rand_dec,
                                      binfile=bins,
                                      weights1=weight_ref,
                                      weights2=weight_rand,
                                      weight_type=weight_type)

    DiR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads,
                                      RA1=ra_int, DEC1=dec_int,
                                      RA2=rand_ra, DEC2=rand_dec,
                                      binfile=bins,
                                      weights1=weight_int,
                                      weights2=weight_rand,
                                      weight_type=weight_type)

    if weight_type == 'pair_product':
        DtDt_ = DtDt['npairs'] * DtDt['weightavg']
        DrDr_ = DrDr['npairs'] * DrDr['weightavg']
        DiDi_ = DiDi['npairs'] * DiDi['weightavg']
        DtDr_ = DtDr['npairs'] * DtDr['weightavg']
        DtR_ = DtR['npairs'] * DtR['weightavg']
        DrR_ = DrR['npairs'] * DrR['weightavg']
        DiR_ = DiR['npairs'] * DiR['weightavg']
        RR_ = RR['npairs'] * RR['weightavg']
        ndata_targ = np.sum(weight_targ)
        ndata_ref = np.sum(weight_ref)
        ndata_int = np.sum(weight_int)
        nrand = np.sum(weight_rand)
    else:
        DtDt_ = DtDt
        DrDr_ = DrDr
        DiDi_ = DiDi
        DtDr_ = DtDr
        DtR_ = DtR
        DrR_ = DrR
        DiR_ = DiR
        RR_ = RR
        ndata_ref = len(ra_ref)
        ndata_targ = len(ra_targ)
        ndata_int = len(ra_int)
        nrand = len(rand_ra)

    #poisson error calculation
    errorDtDt= np.sqrt(DtDt_)
    errorDrDr= np.sqrt(DrDr_)
    errorDiDi= np.sqrt(DiDi_)
    errorDtDr= np.sqrt(DtDr_)
    errorDtR= np.sqrt(DtR_)
    errorDrR= np.sqrt(DrR_)
    errorDiR= np.sqrt(DiR_)
    errorRR= np.sqrt(RR_)


    print("calculating cross correlation")

    # All the pair counts are done, get the angular correlation function
    wtheta_cross = Corrfunc.utils.convert_3d_counts_to_cf(ndata_targ, ndata_ref, nrand, nrand,
                                                          DtDr_, DtR_, DrR_, RR_)


    #propagation of error calculation
    fN1 = np.float64(nrand) / np.float64(ndata_targ)
    fN2 = np.float64(nrand) / np.float64(ndata_ref)
    delDtDr= (fN1 * fN2)/RR_
    delDtR_cross= -(fN1)/RR_
    delDrR= -(fN2)/RR_
    delRR_cross= (fN1 * DtR_ + fN2 * DrR_ - fN1 * fN2 * DtDr_)/np.power(RR_, 2)


    error_cross= np.sqrt(np.power(delDtDr * errorDtDr, 2) + np.power(delDtR_cross * errorDtR, 2) + np.power(delDrR * errorDrR, 2) + np.power(delRR_cross * errorRR, 2))

    print("calculating auto correlation")

    wtheta_auto= Corrfunc.utils.convert_3d_counts_to_cf(ndata_ref, ndata_ref, nrand, nrand,
                                                            DrDr_, DrR_, DrR_, RR_)

    wtheta_int= Corrfunc.utils.convert_3d_counts_to_cf(ndata_int, ndata_int, nrand, nrand,
                                                            DiDi_, DiR_, DiR_, RR_)

    #propagation of error calculation
    fNr = np.float64(nrand) / np.float64(ndata_ref)
    delDrDr= (np.power(fNr, 2))/RR_
    delDrR_auto= -(fNr * 2)/RR_
    delRR_autor= (-np.power(fNr, 2) * DrDr_+ 2 * fNr * DrR_)/np.power(RR_, 2)

    fNi = np.float64(nrand) / np.float64(ndata_int)
    delDiDi= (np.power(fNi, 2))/RR_
    delDiR_auto= -(fNi * 2)/RR_
    delRR_autoi= (-np.power(fNi, 2) * DiDi_+ 2 * fNi * DiR_)/np.power(RR_, 2)

    error_auto= np.sqrt(np.power(delDrDr * errorDrDr, 2) + np.power(delDrR_auto * errorDrR, 2) + np.power(delRR_autor * errorRR,2))

    error_int= np.sqrt(np.power(delDiDi * errorDiDi, 2) + np.power(delDiR_auto * errorDiR, 2) + np.power(delRR_autoi * errorRR,2))

    theta_min = DrDr['thetamin']
    theta_max = DrDr['thetamax']
    theta = (theta_min + theta_max)/2.

    return theta, wtheta_cross, wtheta_auto, wtheta_int, error_cross, error_auto, error_int


def bin_count(indices):
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
    print(f"pixelizing map {len(ra)} {nside=}")
    pix = healpy.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    pix, count = bin_count(pix)
    ra_pix, dec_pix = healpy.pix2ang(nside, pix, nest=True, lonlat=True)
    return ra_pix, dec_pix, count


def compute_wtheta_healpix(rat, dect, rar, decr, rand_ra, rand_dec, ra_int, dec_int, bins, nthreads=1, nside=2048):
    """Compute angular correlation function w(theta) using the pixelized map method"""
    ra_pix_targ, dec_pix_targ, count_targ = pixelize(rat, dect, nside=nside)
    ra_pix_ref, dec_pix_ref, count_ref = pixelize(rar, decr, nside=nside)
    ra_pix_int, dec_pix_int, count_int = pixelize(ra_int, dec_int, nside=nside)
    ra_pix_r, dec_pix_r, count_rand = pixelize(rand_ra, rand_dec, nside=nside)
    theta, wtheta_cross, wtheta_auto, wtheta_int, error_cross, error_auto, error_int = compute_wtheta(ra_pix_targ, dec_pix_targ,
                                                                                                      ra_pix_ref, dec_pix_ref, ra_pix_int, dec_pix_int,
                                                                                                      ra_pix_r, dec_pix_r,
                                                                                                      weight_targ=count_targ, weight_ref=count_ref, weight_int= count_int,
                                                                                                      weight_rand= count_rand, bins=bins, nthreads=nthreads)
    return theta, wtheta_cross, wtheta_auto, wtheta_int, error_cross, error_auto, error_int



def generate_random_catalog(size):
    """ """
    ra_random= np.random.uniform(0, 90, size)
    dec_random = np.random.uniform(0, 90, size)
    ran_cat = np.column_stack((ra_random, dec_random))

    return ran_cat

bins = np.linspace(0.01, 1, 31)

#reference with known redshift: r<19,
#pool for interlopers: r>19
#17419 low redshift
#17418 Euclid Alpha

low_redshift_file = "17419.csv.bz2"
Euclid_Halpha_file = "17418.csv.bz2"
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
int_fraction = targ_cat2.sample(n=Nalpha)
#sum two catalogues
targ_cat= pd.concat([int_fraction, targ_cat1])
#randomize the final catalogue to avoid bias
targ_cat = targ_cat.sample(frac=1, ignore_index=True, random_state=3)
rai=int_fraction['ra_gal']
Nitot=len(rai)


print("generating catalogs")
rat= targ_cat['ra_gal']
dect= targ_cat['dec_gal']
Nt= len(rat)
print("length target catalog is %d" % Nt)

rar=ref_cat['ra_gal']
decr= ref_cat['dec_gal']
Nr= len(rar)
print("length reference catalog is %d" % Nr)

ra_int= int_fraction['ra_gal']
dec_int= int_fraction['dec_gal']
Ni= len(ra_int)
print("length reference catalog is %d" % Ni)

#Random catalogue generation
rand_cat= generate_random_catalog(10*len(rat))
rand_ra, rand_dec = rand_cat[:, 0], rand_cat[:, 1]
rand_N = len(rand_ra)
print("random catalog generated, length is %d" % rand_N)

theta, wtheta_cross, wtheta_auto, wtheta_int, error_cross, error_auto, error_int= compute_wtheta_healpix(rat, dect, rar, decr, rand_ra, rand_dec, ra_int, dec_int, bins)

ratio_wtheta= wtheta_cross/wtheta_auto

error_ratio= np.zeros(len(theta))

for i in range(len(theta)):

    error_ratio[i]= np.absolute(ratio_wtheta[i]) * np.sqrt(np.power(error_cross[i]/wtheta_cross[i],2)+np.power(error_auto[i]/wtheta_auto[i],2))

print("theta=", theta)
print("wtheta_cross=", wtheta_cross)
print("wtheta_auto_ref=", wtheta_auto)
print("wtheta_auto_int=", wtheta_int)
print("ratio=", ratio_wtheta)
print("error_cross=", error_cross)
print("error_auto=", error_auto)
print("error_int=", error_int)
print("error_ratio=", error_ratio)

#integrated omega
plt.figure(figsize=(10, 6))
plt.errorbar(theta, wtheta_cross, yerr= error_cross, label= 'cross correlation', fmt='o', color= 'blue')
plt.errorbar(theta, wtheta_auto, yerr= error_auto, label='auto correlation', fmt='o', color= 'green')
plt.yscale('log')
plt.title('Cross and auto correlations')
plt.xlabel('theta')
plt.ylabel('Correlation')

# Show the plot
plt.show()

plt.plot(theta, ratio_wtheta, 'o', alpha=0.6, color='blue')

# Adding titles and labels
plt.title('interloper fraction')
plt.xlabel('theta')
plt.ylabel('w_x/ w_ref')
plt.legend()

# Show the plot
plt.show()
