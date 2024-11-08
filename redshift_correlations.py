import numpy as np
import healpy
import pandas as pd
import Corrfunc
from matplotlib import pyplot as plt

def read_target_catalog_Euclid(filename):
    """ """
    #high redshift catalog from Euclid
    print(f"reading high redshift sample {filename}")
    catB = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    raB = catB['ra_gal']
    decB = catB['dec_gal']
    # cut the octant (a few galaxies are outside)
    selB1 = (raB > 0)&(raB < 90)&(decB > 0)&(decB < 90)
    catB = catB[selB1]
    #very bright galaxies
    highredshift= catB['observed_redshift_gal']
    selB = highredshift > 0.9
    catB = catB[selB]
    #The catalog is not randomized here because it will be randomized later
    return catB

def read_target_catalog_interloper(filename):
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
    #The catalog is not randomized here because it will be randomized later
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

def generate_random_catalog(size):
    """ The routine generates a random catalog """
    ra_rand= np.random.uniform(0, 90, size)
    z = np.random.uniform(0, 1, size)
    dec_rand = 90-np.arccos(z) * 180/np.pi
    return ra_rand, dec_rand

def compute_fixed_pairs(ra1, dec1, rand_ra, rand_dec, bins=None, weight=None, weight_rand= None, nthreads=1):
    '''Computing fixed pairs, auto correlation of the random catalog and cross correlation of the Target(interlopers+Euclid) and
       random catalog'''

    weight_type = None

    if weight is not None:
        weight_type = 'pair_product'
        print(f"data weights for target pair {weight.min()} {weight.max()}")
        print(f"rand weights for random pair {weight_rand.min()} {weight_rand.max()}")

    autocorr = 1
    crosscorr = 0

    print("Computing fixed pairs")

    # Cross pair counts in DtR
    DR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                       RA1=ra1, DEC1=dec1, RA2=rand_ra, DEC2=rand_dec, weights1=weight, weights2=weight_rand,
                                       weight_type=weight_type)


    # Auto pair counts in RR
    RR = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads, RA1=rand_ra, DEC1=rand_dec,
                                      binfile=bins, weights1=weight_rand, weight_type=weight_type)

    #Normalizing pair counts, the weights will be calculated later for the correlations.
    #A good check is keeping an eye on what data weights says for the fixed pair section and the correlations sections

    if weight_type == 'pair_product':
        DR_ = DR['npairs'] * DR['weightavg']
        RR_ = RR['npairs'] * RR['weightavg']
    else:
        DR_ = DR
        RR_ = RR

    return DR_, RR_


def compute_wtheta_cross(ra_targ, dec_targ, ra_ref, dec_ref, rand_ra, rand_dec, DtR_, RR_,  bins=None, weight_targ=None, weight_ref=None, weight_rand= None, nthreads=1):
    """Computing angular cross correlation function for a pixelized map """

    weight_type = None
    if weight_ref is not None:
        weight_type = 'pair_product'
        print(f"data weights for target pair {weight_targ.min()} {weight_targ.max()}")
        print(f"data weights for reference pair {weight_ref.min()} {weight_ref.max()}")
        print(f"rand weights for random pair {weight_rand.min()} {weight_rand.max()}")


    autocorr = 1
    crosscorr = 0

    # Cross pair counts in DtDr
    DtDr = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                        RA1=ra_targ, DEC1=dec_targ, RA2=ra_ref, DEC2=dec_ref, weights1=weight_targ, weights2=weight_ref,
                                        weight_type=weight_type)

    # Cross pair counts in DrR

    DrR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                       RA1=ra_ref, DEC1=dec_ref, RA2=rand_ra, DEC2=rand_dec, weights1=weight_ref, weights2=weight_rand,
                                       weight_type=weight_type)

    #Normalizing pair counts


    if weight_type == 'pair_product':
        DtDr_ = DtDr['npairs'] * DtDr['weightavg']
        DrR_ = DrR['npairs'] * DrR['weightavg']
        ndata_targ= np.sum(weight_targ)
        ndata_ref = np.sum(weight_ref)
        nrand = np.sum(weight_rand)

    else:
        DtDr_ = DtDr
        DrR_ = DrR
        ndata_targ = len(ra_targ)
        ndata_ref = len(ra_ref)
        nrand = len(rand_ra)


    #poisson error calculation
    errorDtDr= np.sqrt(DtDr_)
    errorDtR= np.sqrt(DtR_)
    errorDrR= np.sqrt(DrR_)
    errorRR= np.sqrt(RR_)

    # All the pair counts are done, get the angular correlation function
    wtheta_cross = Corrfunc.utils.convert_3d_counts_to_cf(ndata_targ, ndata_ref, nrand, nrand,
                                                          DtDr_, DtR_, DrR_, RR_)

    #propagation of error calculation using differential formula, not the main issue for now

    fN1 = np.float64(nrand) / np.float64(ndata_targ)
    fN2 = np.float64(nrand) / np.float64(ndata_ref)
    delDtDr= (fN1 * fN2)/RR_
    delDtR_cross= -(fN1)/RR_
    delDrR= -(fN2)/RR_
    delRR_cross= (fN1 * DtR_ + fN2 * DrR_ - fN1 * fN2 * DtDr_)/np.power(RR_, 2)


    error_cross= np.sqrt(np.power(delDtDr * errorDtDr, 2) + np.power(delDtR_cross * errorDtR, 2) + np.power(delDrR * errorDrR, 2) + np.power(delRR_cross * errorRR, 2))

    #theta calculation
    theta_min = DtDr['thetamin']
    theta_max = DtDr['thetamax']
    theta = (theta_min + theta_max)/2.
    return theta, wtheta_cross, error_cross


def compute_wtheta_auto(ra, dec, rand_ra, rand_dec, RR_, bins=None, weight=None, weight_rand=None, nthreads=1):
    """Computing angular auto correlation function for a pixelized map, the names are general because both the interloper
       fraction and the reference catalog will go through"""

    weight_type = None
    if weight is not None:
        weight_type = 'pair_product'
        print(f"data weights for catalog {weight.min()} {weight.max()}")
        print(f"rand weights for random {weight_rand.min()} {weight_rand.max()}")


    autocorr = 1
    crosscorr = 0

    # Cross pair counts in DR
    DR = Corrfunc.mocks.DDtheta_mocks(crosscorr, nthreads, binfile=bins,
                                       RA1=ra, DEC1=dec, RA2=rand_ra, DEC2=rand_dec, weights1=weight, weights2=weight_rand,
                                       weight_type=weight_type)


    # Auto pairs counts in DD
    DD = Corrfunc.mocks.DDtheta_mocks(autocorr, nthreads, RA1=ra, DEC1=dec,
                                        binfile=bins, weights1=weight, weight_type=weight_type)

    #Normalizing pair counts

    if weight_type == 'pair_product':
        DD_ = DD['npairs'] * DD['weightavg']
        DR_ = DR['npairs'] * DR['weightavg']
        ndata = np.sum(weight)
        nrand = np.sum(weight_rand)
    else:
        DD_ = DD
        DR_ = DR
        ndata = len(ra)
        nrand = len(rand_ra)

    #poisson error calculation
    errorDR= np.sqrt(DR_)
    errorDD= np.sqrt(DD_)
    errorRR= np.sqrt(RR_)

    # All the pair counts are done, get the angular correlation function
    wtheta_auto= Corrfunc.utils.convert_3d_counts_to_cf(ndata, ndata, nrand, nrand,
                                                        DD_, DR_, DR_, RR_)

    #propagation of error calculation with differential formula

    fN = np.float64(nrand) / np.float64(ndata)
    delDD= (np.power(fN, 2))/RR_
    delDR_auto= -(fN * 2)/RR_
    delRR_auto= (-np.power(fN, 2) * DD_+ 2 * fN * DR_)/np.power(RR_, 2)


    error_auto= np.sqrt(np.power(delDD * errorDD, 2) + np.power(delDR_auto * errorDR, 2) + np.power(delRR_auto * errorRR,2))

    #theta calculation

    theta_min = DD['thetamin']
    theta_max = DD['thetamax']
    theta = (theta_min + theta_max)/2.

    return theta, wtheta_auto, error_auto

def bin_count(indices):
    """Return number count for each bin index

    Parameters:
    -----------
    indices : array
      integer bin indices

    Returns:
    --------
    integer array pix_list,
    float array count
    """
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
    """Compute pixelized map for array of points.

    Parameters
    ----------
    ra : array
      RA coordinate
    dec : array
      Dec coordinate
    nside : int
      healpix nside parameters
    """
    print(f"pixelizing map {len(ra)} {nside=}")
    pix = healpy.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    pix, count = bin_count(pix)
    ra_pix, dec_pix = healpy.pix2ang(nside, pix, nest=True, lonlat=True)
    return ra_pix, dec_pix, count

def compute_fixed_healpix(ra, dec, ra_r, dec_r, bins, nthreads=1, nside=2048):
    """Compute  pixel map and pair counts for fixed pairs"""
    ra_pix, dec_pix, count = pixelize(ra, dec, nside=nside)
    ra_pix_r, dec_pix_r, count_r = pixelize(ra_r, dec_r, nside=nside)
    DtR_, RR_ = compute_fixed_pairs(ra_pix, dec_pix,
                                    ra_pix_r, dec_pix_r,
                                    weight=count, weight_rand=count_r,
                                    bins=bins, nthreads=nthreads)
    return DtR_, RR_

def compute_cross_healpix(ra_targ, dec_targ, ra_ref, dec_ref, ra_rand, dec_rand, RR_, DtR_, bins, nthreads=1, nside=2048):
    """Compute angular cross correlation function w(theta) using the pixelized map method"""
    ra_pix_targ, dec_pix_targ, count_targ = pixelize(ra_targ, dec_targ, nside=nside)
    ra_pix_ref, dec_pix_ref, count_ref = pixelize(ra_ref, dec_ref, nside=nside)
    ra_pix_rand, dec_pix_rand, count_rand = pixelize(rand_ra, rand_dec, nside=nside)
    theta, wtheta_cross, wtheta_cross_error = compute_wtheta_cross(ra_pix_targ, dec_pix_targ, ra_pix_ref, dec_pix_ref,
                                                            ra_pix_rand, dec_pix_rand, RR_, DtR_,
                                                            weight_targ=count_targ, weight_ref=count_ref, weight_rand= count_rand,
                                                            bins=bins, nthreads=nthreads)

    return theta, wtheta_cross, wtheta_cross_error

def compute_auto_healpix(ra, dec, ra_r, dec_r, RR_, bins, nthreads=1, nside=2048):
    """Compute pixel map and auto correlation"""
    ra_pix, dec_pix, count = pixelize(ra, dec, nside=nside)
    ra_pix_r, dec_pix_r, count_r = pixelize(ra_r, dec_r, nside=nside)
    theta, wtheta_auto, wtheta_auto_error = compute_wtheta_auto(ra_pix, dec_pix,
                                    ra_pix_r, dec_pix_r, RR_,
                                    weight=count, weight_rand=count_r,
                                    bins=bins, nthreads=nthreads)

    return theta, wtheta_auto, wtheta_auto_error

#setup the bins for the angula count. Corrfunc suggests to use n+1 as number of bins. Hence, I used 11.

bins = np.linspace(0.01, 1, 11)

#Begin reading the catalogs

#reference with known redshift: r<19,
#pool for interlopers: r>19
#17419 low redshift
#17418 Euclid Alpha

low_redshift_file = "17419.csv.bz2"
Euclid_Halpha_file = "17418.csv.bz2"
targ_cat_Euclid = read_target_catalog_Euclid(Euclid_Halpha_file)
targ_cat_interlopers = read_target_catalog_interloper(low_redshift_file)
ref_cat = read_reference_catalog(low_redshift_file)

#Begin creating the catalog for the analysis
'''
    The equation we followed:

                                   (number of interlopers in the catalog)
    intfraction= 0.1 = ------------------------------------------------------------
                        (number of interlopers in the catalog) + (Euclid galaxies)
'''

intfraction= 0.1

print("selecting interlopers fraction. Chosen is {:.2f}".format(intfraction))

raEuclid= targ_cat_Euclid['ra_gal']
NEuclid= len(raEuclid)
Ninterloper_fraction= int(((intfraction)/(1-intfraction))*NEuclid)

#select random elements from catalog with the interlopers to create the interloper fraction

int_fraction = targ_cat_interlopers.sample(n=Ninterloper_fraction)

#sum of Euclid + interlopers to create the target catalog

targ_cat= pd.concat([int_fraction, targ_cat_Euclid])

#randomize the final catalogue to avoid bias

targ_cat = targ_cat.sample(frac=1, ignore_index=True, random_state=3)


#Begin the generation of the other catalogs

print("generating catalogs")
ra_targ= targ_cat['ra_gal']
dec_targ= targ_cat['dec_gal']
Nt= len(ra_targ)
print("length target catalog is %d" % Nt)

ra_ref=ref_cat['ra_gal']
dec_ref= ref_cat['dec_gal']
Nr= len(ra_ref)
print("length reference catalog is %d" % Nr)

#Random catalogue generation
rand_ra, rand_dec= generate_random_catalog(10*Nt)
N_rand = len(rand_ra)
print("random catalog generated, length is %d" % N_rand)

#Check the interloper fraction

ra_int= int_fraction['ra_gal']
N_int= len(ra_int)

fraction= N_int/Nt

print("The interloper fraction achieved is {:.2f}".format(fraction))


#Generating arrays for calculations

'''
The aim is to go from angular to redshift correlation. Hence we will use the formula (S stands for summation)

            S (w(theta) * theta^(-0.8))
   w(z)=   -----------------------------
               S (theta^(-0.8))

'''

#Redshift arrays to create the bins
redshift_ref = ref_cat['observed_redshift_gal']
redshift_int = int_fraction['observed_redshift_gal']

#Redshift bins
redshift_bins=np.linspace(0, 0.48275862, 29)
increment= redshift_bins[1] - redshift_bins[0]

#Array for the sum of the thetasin the redshift bins for the final formula
thetaDint= np.zeros(len(redshift_bins), dtype=float)
thetaDcross= np.zeros(len(redshift_bins), dtype=float)
thetaDauto= np.zeros(len(redshift_bins), dtype=float)

#Omega is the resulting integration, wtheta would be incorret. OmegaN is the numerator for the final formula

#Cross correlation
omegaN_cross= np.zeros(len(redshift_bins), dtype=float)
omega_cross= np.zeros(len(redshift_bins), dtype=float)
error_cross_squared= np.zeros(len(redshift_bins), dtype=float)
error_cross= np.zeros(len(redshift_bins), dtype=float)

#Auto correlation
omegaN_auto= np.zeros(len(redshift_bins), dtype=float)
omega_auto= np.zeros(len(redshift_bins), dtype=float)
error_auto_squared= np.zeros(len(redshift_bins), dtype=float)
error_auto= np.zeros(len(redshift_bins), dtype=float)

#Interlopers auto correlation
omegaN_int=np.zeros(len(redshift_bins), dtype=float)
omega_int=np.zeros(len(redshift_bins), dtype=float)
error_int_squared=np.zeros(len(redshift_bins), dtype=float)
error_int=np.zeros(len(redshift_bins), dtype=float)

#This array will contain the number of interlopers in a redshift bin divided by the whole target catalog

'''
                            Interlopers (z)
interlopers= -----------------------------------------------
              Interloper_fraction + Euclid (target acatlog)

'''

interlopers= np.zeros(len(redshift_bins), dtype=float)

#The following array will contain the error of the ratio between cross and auto correlation of the reference catalog

error_ratio_omega= np.zeros(len(redshift_bins), dtype=float)


#Correlation function calculations and binning

print("Fixed pairs count started")

DtR_, RR_= compute_fixed_healpix(ra_targ, dec_targ, rand_ra, rand_dec, bins)

print("Fixed pairs count is over")

#Cross correlation

print("Starting cross correlations-----------------------------------------------------------------------")

for i in range(len(redshift_bins)):

    #First we create the bin in the redshift data

    sel_ref = (redshift_ref>= redshift_bins[i])&(redshift_ref < redshift_bins[i]+increment)

    #Next we chose the galaxies in the bin by cutting the catalog
    ref_cat_cut= ref_cat[sel_ref]

    #Finally we select the locations of the galaxies
    rar_bin=ref_cat_cut['ra_gal']
    decr_bin= ref_cat_cut['dec_gal']
    Nr_bin= len(rar_bin)
    print("length reference catalog at redshift %f is %d" % (redshift_bins[i], Nr_bin))

    #Calculating the correlation
    #We pass the data of the target catalog first, and then of the reference (in the bin) and random. Also, we pass fixed pairs
    #and the bins of the angle

    theta, wtheta_cross, error_cross_bin= compute_cross_healpix(ra_targ, dec_targ, rar_bin, decr_bin, rand_ra, rand_dec, DtR_, RR_, bins)

    print("At redshift %f the results for the cross correlation are:" % (redshift_bins[i]))
    print("theta=", theta)
    print("wtheta_cross=", wtheta_cross)
    print("error= ", error_cross_bin)

    #It is possible to visualize a plot for every bin for better analysis

    #plt.figure(figsize=(10, 6))
    #plt.errorbar(theta, wtheta_cross, yerr= error_cross_bin, fmt='o', color= 'blue')
    #plt.yscale('log')
    #plt.title('Angular correlation between redshift bins of the reference catalog and the target catalog')
    #plt.xlabel(r'$\theta$ (degrees)')
    #plt.ylabel(r'$\omega(\theta)$')

    #plt.show()

    #Now we begin the calculations for the redshift correlation

    for j in range(len(theta)):

        #Every iteration we sum the angles and the data for the numerator

        thetaDcross[i] += np.power(theta[j], -0.8)
        omegaN_cross[i] += (wtheta_cross[j] * np.power(theta[j], -0.8))

        error_cross_squared[i] += np.power(np.power(theta[j], -0.8) * error_cross_bin[j], 2)

    error_cross[i]= np.sqrt(error_cross_squared[i])/thetaDcross[i]

    omega_cross[i]= omegaN_cross[i]/thetaDcross[i]

#Auto correlation

print("Starting auto correlations for reference catalog------------------------------------------------------------")

for i in range(len(redshift_bins)):

    #First we create the bin in the redshift data

    sel_ref = (redshift_ref>= redshift_bins[i])&(redshift_ref < redshift_bins[i]+increment)

    #Next we chose the galaxies in the bin by cutting the catalog
    ref_cat_cut= ref_cat[sel_ref]

    #Finally we select the locations of the galaxies
    rar_bin=ref_cat_cut['ra_gal']
    decr_bin= ref_cat_cut['dec_gal']
    Nr_bin= len(rar_bin)
    print("length reference catalog at redshift %f is %d" % (redshift_bins[i], Nr_bin))

    #Calculating the correlation
    #We pass the data of the reference catalog first, and then of the random. Also, we pass fixed pairs
    #and the bins of the angle

    theta, wtheta_auto, error_auto_bin= compute_auto_healpix(rar_bin, decr_bin, rand_ra, rand_dec, RR_, bins)

    print("At redshift %f the results for the auto correlation are:" % (redshift_bins[i]))
    print("theta=", theta)
    print("wtheta_auto=", wtheta_auto)
    print("error= ", error_auto_bin)

    #It is possible to visualize a plot for every bin for better analysis

    #plt.figure(figsize=(10, 6))
    #plt.errorbar(theta, wtheta_auto, yerr= error_auto_bin, fmt='o', color= 'blue')
    #plt.yscale('log')
    #plt.title('Angular correlation between redshift bins of the reference catalog')
    #plt.xlabel(r'$\theta$ (degrees)')
    #plt.ylabel(r'$\omega(\theta)$')

    #plt.show()

    #Now we begin the calculations for the redshift correlation

    for j in range(len(theta)):

        #Every iteration we sum the angles and the data for the numerator

        thetaDauto[i] += np.power(theta[j], -0.8)
        omegaN_auto[i] += (wtheta_auto[j] * np.power(theta[j], -0.8))

        error_auto_squared[i] += np.power(np.power(theta[j], -0.8) * error_auto_bin[j], 2)

    error_auto[i]= np.sqrt(error_auto_squared[i])/thetaDauto[i]

    omega_auto[i]= omegaN_auto[i]/thetaDauto[i]

#Auto correlation interlopers

print("Starting auto correlations for interloper fraction------------------------------------------------------------")

for i in range(len(redshift_bins)):

    #First we create the bin in the redshift data

    sel_int = (redshift_int>= redshift_bins[i])&(redshift_int < redshift_bins[i]+increment)

    #Next we chose the galaxies in the bin by cutting the catalog
    int_cat_cut= int_fraction[sel_int]

    #Finally we select the locations of the galaxies
    rai_bin=int_cat_cut['ra_gal']
    deci_bin= int_cat_cut['dec_gal']
    Ni_bin= len(rai_bin)
    interlopers[i]= Ni_bin / Nt
    print("length interlopers fraction at redshift %f is %d" % (redshift_bins[i], Ni_bin))

    #Calculating the correlation
    #We pass the data of the interloper catalog first, and then of the random. Also, we pass fixed pairs
    #and the bins of the angle

    theta, wtheta_int, error_int_bin= compute_auto_healpix(rai_bin, deci_bin, rand_ra, rand_dec, RR_, bins)

    print("At redshift %f the results for the auto correlation are:" % (redshift_bins[i]))
    print("theta=", theta)
    print("wtheta_int=", wtheta_int)
    print("error= ", error_int_bin)

    #It is possible to visualize a plot for every bin for better analysis

    #plt.figure(figsize=(10, 6))
    #plt.errorbar(theta, wtheta_int, yerr= error_int_bin, fmt='o', color= 'blue')
    #plt.yscale('log')
    #plt.title('Angular correlation between redshift bins of the interloper catalog')
    #plt.xlabel(r'$\theta$ (degrees)')
    #plt.ylabel(r'$\omega(\theta)$')

    #plt.show()

    #Now we begin the calculations for the redshift correlation

    for j in range(len(theta)):

        #Every iteration we sum the angles and the data for the numerator

        thetaDint[i] += np.power(theta[j], -0.8)
        omegaN_int[i] += (wtheta_int[j] * np.power(theta[j], -0.8))

        error_int_squared[i] += np.power(np.power(theta[j], -0.8) * error_int_bin[j], 2)

    error_int[i]= np.sqrt(error_int_squared[i])/thetaDint[i]

    omega_int[i]= omegaN_int[i]/thetaDint[i]


#Now we calculate the ratio between the cross correlation and the auto correlation of the reference catalog

ratio_omega= omega_cross/omega_auto

for i in range(len(redshift_bins)):

    error_ratio_omega[i]= np.absolute(ratio_omega[i]) * np.sqrt(np.power(error_cross[i]/omega_cross[i],2)+np.power(error_auto[i]/omega_auto[i],2))

#Final results are printed on a txt file for further analysis

results = "redshift_correlations.txt"

with open(results, 'w') as f:
    f.write("Results for redshift correlations\n")
    f.write("omega_cross = " + repr(omega_cross) + "\n")
    f.write("omega_auto = " + repr(omega_auto) + "\n")
    f.write("omega_int = " + repr(omega_int) + "\n")
    f.write("ratio_omega = " + repr(ratio_omega) + "\n")
    f.write("error_cross = " + repr(error_cross) + "\n")
    f.write("error_auto = " + repr(error_auto) + "\n")
    f.write("error_int = " + repr(error_int) + "\n")
    f.write("error_ratio_omega = " + repr(error_ratio_omega) + "\n")
    f.write("interloper_fraction = " + repr(interlopers) + "\n")
    f.write(f"Interloper fraction: {np.sum(interlopers):.5f}\n")
    f.write(f"Interloper fraction detected: {np.sum(ratio_omega):.5f}\n")
    f.write(f"Error analysis: {np.sum(interlopers)-np.sum(ratio_omega):.5f}\n")
    f.write(f"With an uncertainty of: {np.sum(error_ratio_omega):.5f}\n")

print(f"Results saved to {results}")

#Plot analysis

#plot of the correlations

plt.figure(figsize=(10, 6))
plt.errorbar(redshift_bins, omega_cross, yerr= error_cross, fmt='o', color= 'blue')
plt.errorbar(redshift_bins, omega_auto, yerr= error_auto, fmt='o', color= 'green')
plt.yscale('log')
plt.title('Auto and Cross redshift correlations of the reference catalog across Redshift bins')
plt.xlabel('Redshift Bins')
plt.ylabel(r'$\omega(z)$')

plt.show()

#plot of the ratio related with the interloper fraction

plt.plot(redshift_bins, interlopers, 'o', label='Interlopers fraction normalized', alpha=0.6, color='blue')
plt.plot(redshift_bins, ratio_omega, 'o', label='Ratio of cross and auto redshift correlations', color='green')
plt.title('Interloper detection results')
plt.xlabel('Redshift Bins')
plt.ylabel(r'$\frac{\omega(z)_{\text{cross}}}{\omega(z)_{\text{auto}}}$')
plt.legend()

plt.show()


