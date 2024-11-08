import numpy as np
import pandas as pd
from math import sqrt, pi, sin, sinh
from matplotlib import pyplot as plt

def read_target_catalog(filename):
    """ """
    #high redshift catalog
    print(f"reading high redshift sample {filename}")
    catB = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    raB = catB['ra_gal']
    decB = catB['dec_gal']
    #cut the octant (a few galaxies are outside)
    selB1 = (raB > 0)&(raB < 90)&(decB > 0)&(decB < 90)
    catB = catB[selB1]
    return catB


def read_target_catalogB(filename):
    """ """
    #high redshift catalog
    print(f"reading high redshift sample {filename}")
    catB = pd.read_csv(filename, sep=",", comment='#', na_values=r'\N', compression='bz2')
    raB = catB['ra_gal']
    decB = catB['dec_gal']
    #cut the octant (a few galaxies are outside)
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
    #cut the octant (a few galaxies are outside)
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
    #cut the octant (a few galaxies are outside)
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
    """ """
    ra_random= np.random.uniform(0, 90, size)
    dec_random = np.random.uniform(0, 90, size)
    #dec_random = 90 - (180 / np.pi) * np.arccos(y)
    ran_cat = np.column_stack((ra_random, dec_random))
    #sel = (ra_random > 0)&(ra_random < 10)&(dec_random > 0)&(dec_random < 10)
    #ran_cat= ran_cat[sel]

    return ran_cat

def comoving_volume(z1, z2, Nt):

    '''calculates comoving volume based on redshift assuming a flat universe'''
    # Constants
    H0 = 69.6  # Hubble constant in km/s/Mpc
    WM = 0.315  # Omega matter
    WV = 1.0 - WM  # Omega vacuum (assuming a flat universe)
    c = 299792.458  # Speed of light in km/s


    # Derived constants
    Tyr = 977.8  # Coefficient for converting 1/H0 into Gyr
    WR = 4.165E-5 / ((H0 / 100.0) ** 2)  # Omega radiation
    WK = 1.0 - WM - WR - WV  # Omega curvature

    # Initialize variables
    az1 = 1.0 / (1 + z1)
    az2= 1.0 / (1+z2)
    DCMR1 = 0.0  # Comoving radial distance in units of c/H0
    DCMR2= 0.0

    # Calculate the comoving radial distance (DCMR)
    n = 1000  # Number of integration points
    for i in range(n):
        a1 = az1 + (1 - az1) * (i + 0.5) / n
        adot1 = sqrt(WK + (WM / a1) + (WR / (a1 * a1)) + (WV * a1 * a1))
        DCMR1 += 1.0 / (a1 * adot1)

        a2 = az2 + (1 - az2) * (i + 0.5) / n
        adot2 = sqrt(WK + (WM / a2) + (WR / (a2 * a2)) + (WV * a2 * a2))
        DCMR2 += 1.0 / (a2 * adot2)

    DCMR1 = (1.0 - az1) * DCMR1 / n
    DCMR_Mpc1 = (c / H0) * DCMR1  # Convert to Mpc

    DCMR2 = (1.0 - az2) * DCMR2 / n
    DCMR_Mpc2 = (c / H0) * DCMR2  # Convert to Mpc

    # Calculate the comoving volume (VCM)
    ratio1 = 1.0
    ratio2 = 1.0
    x1 = sqrt(abs(WK)) * DCMR1
    x2 = sqrt(abs(WK)) * DCMR2

    if x1 > 0.1:
        if WK > 0:
            ratio1 = 0.5 * (exp(x1) - exp(-x1)) / x1
        else:
            ratio1 = sin(x1) / x1
    else:
        y1 = x1 * x1
        if WK < 0:
            y1 = -y1
        ratio1 = 1.0 + y1 / 6.0 + y1 * y1 / 120.0

    if x2 > 0.1:
        if WK > 0:
            ratio2 = 0.5 * (exp(x2) - exp(-x2)) / x2
        else:
            ratio2 = sin(x2) / x2
    else:
        y2 = x2 * x2
        if WK < 0:
            y2 = -y2
        ratio2 = 1.0 + y2 / 6.0 + y2 * y2 / 120.0

    h= H0/100

    VCM1 = ratio1 * DCMR1 * DCMR1 * DCMR1 / 3.0
    V_Mpc1 = ((( 4.0 * pi * ((0.001 * c / H0) ** 3) * VCM1 )/8) * h**3) * 1e9 # Comoving volume in Mpc^3/h^3

    VCM2 = ratio2 * DCMR2 * DCMR2 * DCMR2 / 3.0
    V_Mpc2 = ((( 4.0 * pi * ((0.001 * c / H0) ** 3) * VCM2 )/8) * h**3) * 1e9 # Comoving volume in Mpc^3/h^3

    V_Mpc= (V_Mpc2-V_Mpc1)

    print(f'The comoving volume within redshift z={z1} and z= {z2} for an octant of the sky is {V_Mpc:.3f} Mpc^3/h^3.')

    rho= Nt / (V_Mpc)

    return rho


#reference with known redshift: r<19,
#pool for interlopers: r>19
#17419 low redshift
#17418 Euclid Alpha

low_redshift_file = "17419.csv.bz2"
Euclid_Halpha_file = "17418.csv.bz2"
targ_cat1 = read_target_catalogB(Euclid_Halpha_file)
targ_cat2 = read_target_catalogA(low_redshift_file)
ref_cat = read_reference_catalog(low_redshift_file)
euclid= read_target_catalog(Euclid_Halpha_file)

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
print(len(targ_cat))

redshift_int = targ_cat['observed_redshift_gal']
redshift_ref = ref_cat['observed_redshift_gal']
redshift_targ = euclid['observed_redshift_gal']

redshift_bins= np.linspace(0, 0.5, 30)
increment1= redshift_bins[1]
euclid_bins= np.linspace(0.87, 1.10, 10)
increment2= redshift_bins[1]-redshift_bins[0]
galNi= np.zeros(len(redshift_bins))
galNr= np.zeros(len(redshift_bins))
galNt= np.zeros(len(euclid_bins))
rhoint= np.zeros(len(redshift_bins))
rhoref= np.zeros(len(redshift_bins))
rhotarg_whole= np.zeros(len(euclid_bins))

for i in range(len(redshift_bins)):

    seli = (redshift_int> (redshift_bins[i]))&(redshift_int < redshift_bins[i]+increment1)
    targ_cat_cut= targ_cat[seli]
    rai=targ_cat_cut['ra_gal']
    deci= targ_cat_cut['dec_gal']
    Ni= len(rai)
    galNi[i]= Ni
    print("length interloper catalog at redshift %f is %d" % (redshift_bins[i], Ni))
    rhoint[i]= comoving_volume(redshift_bins[i], redshift_bins[i]+increment1, galNi[i])

for i in range(len(redshift_bins)):

    selr = (redshift_ref> (redshift_bins[i]))&(redshift_ref < redshift_bins[i]+increment1)
    ref_cat_cut= ref_cat[selr]
    rar=ref_cat_cut['ra_gal']
    decr= targ_cat_cut['dec_gal']
    Nr= len(rar)
    galNr[i]= Nr
    print("length reference catalog at redshift %f is %d" % (redshift_bins[i], Nr))
    rhoref[i]= comoving_volume(redshift_bins[i], redshift_bins[i]+increment1, galNr[i])

for i in range(len(euclid_bins)):

    selt = (redshift_targ> (euclid_bins[i]))&(redshift_targ < euclid_bins[i]+increment2)
    targ_cat_cut= euclid[selt]
    rat=targ_cat_cut['ra_gal']
    dect= targ_cat_cut['dec_gal']
    Nt= len(rat)
    galNt[i]= Nt
    print("length target catalog at redshift %f is %d" % (euclid_bins[i], Nt))
    rhotarg_whole[i]= comoving_volume(euclid_bins[i], euclid_bins[i]+increment2, galNt[i])

print("Number density in N/(Mpc^3/h^3) of interloper pool=",rhoint)
print("Number density in N/(Mpc^3/h^3) of reference catalog=",rhoref)
print("Number density in N/(Mpc^3/h^3) of Euclid catalog=",rhotarg_whole)

# Plotting
plt.figure(figsize=(12, 8))

# Line plots instead of scatter plots ('o' removed)
plt.plot(redshift_bins, rhoint, label=r'Interlopers Fraction Density', color='blue')
plt.plot(redshift_bins, rhoref, label=r'Reference Catalog Density', color='green')
plt.plot(euclid_bins, rhotarg_whole, label=r'Euclid Mock Catalog Sample Density', color='purple')

# Adding titles and labels
plt.title(r'Number densities of the catalogs in the analysis')
plt.xlabel(r'True Redshift Bins')
plt.ylabel(r'$\frac{\text{N}}{\text{Mpc}^3}$', fontsize=19)
#scaled with h^3
plt.yscale('log')
plt.legend()

# Show the plot
plt.show()







