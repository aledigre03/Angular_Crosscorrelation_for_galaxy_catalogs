import numpy as np
import pandas as pd
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf



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
    #dec_random = 90 - (180 / np.pi) * np.arccos(y)
    ran_cat = np.column_stack((ra_random, dec_random))
    #sel = (ra_random > 0)&(ra_random < 10)&(dec_random > 0)&(dec_random < 10)
    #ran_cat= ran_cat[sel]
    
    return ran_cat

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
print("selecting interlopers fraction. Chosen is %f" % topN)
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
RA1= targ_cat['ra_gal']
DEC1= targ_cat['dec_gal']
N1= len(RA1) 
print("length target catalog is %d" % N1)

RAB=ref_cat['ra_gal']
DECB= ref_cat['dec_gal']
NB= len(RAB)
print("length reference catalog is %d" % NB)

#Random catalogue generation
ran_cat= generate_random_catalog(10*len(RA1))
rand_RA, rand_DEC = ran_cat[:, 0], ran_cat[:, 1]
rand_N = len(rand_RA)
print("random catalog generated, length is %d" % rand_N)

# Setup the bins
nbins = 20
bins = np.linspace(0.1, 5.0, nbins + 1) # note the +1 to nbins

# Number of threads to use
nthreads = 2

# Cross pair counts in D1R
autocorr=0
D1R_counts = DDtheta_mocks(autocorr, nthreads, bins,
                          RA1, DEC1,
                          RA2=rand_RA, DEC2=rand_DEC)
print("D1R cross correlation completed")

# Cross pair counts in DBR
autocorr=0
DBR_counts = DDtheta_mocks(autocorr, nthreads, bins,
                          RAB, DECB,
                          RA2=rand_RA, DEC2=rand_DEC)
print("DBR cross correlation completed")

# Cross pair counts in D1DB
autocorr=0
D1DB_counts = DDtheta_mocks(autocorr, nthreads, bins,
                          RA1, DEC1,
                          RA2=RAB, DEC2=DECB)
print("D1DB cross correlation completed")

# Auto pairs counts in D1D1
autocorr=1
D1D1_counts = DDtheta_mocks(autocorr, nthreads, bins,
                            RA1, DEC1)
print("D1D1 auto correlation completed")

# Auto pair counts in RR
autocorr=1
RR_counts = DDtheta_mocks(autocorr, nthreads, bins,
                          rand_RA, rand_DEC)
print("RR auto correlation completed")

# All the pair counts are done, get the angular correlation function
wtheta_cross = convert_3d_counts_to_cf(N1, NB, rand_N, rand_N,
                                       D1DB_counts, D1R_counts,
                                       DBR_counts, RR_counts)

print("wtheta_cross=", wtheta_cross)


wtheta_auto= convert_3d_counts_to_cf(N1, N1, rand_N, rand_N, 
                                     D1D1_counts, D1R_counts,
                                     D1R_counts, RR_counts)

print("wtheta_auto=", wtheta_auto)

