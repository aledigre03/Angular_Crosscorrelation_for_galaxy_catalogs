# Cross Correlation methods to study the interloper fraction on Euclid catalogs
The files in this repository were created to test and perform angular correlation between galaxy catalogs. The main use is to find interlopers in catalogs produced by the Euclid mission.

-numberdensity.py computes the number density of galaxy catalogs in a range of redshifts in units of N/(Mpc^3) normalized by the Hubble constant h^3

-redshift_correlations.py performs the contruction of the catalags used in the analysis and calculates the cross-correlation and the auto-correlations of the reference catalog and the interloper fraction. It will print out all the details of angular and redshift correlations with the adequate graphs. It will be possible to determine the fraction of interlopers detected.

-photocorr.py is a prototype of the method implemented in redshift_correlations.py but for photometric surveys like Euclid

-galaxy_bias.py performs the calculation of the cosmological biases derived from the data produced by the correlation. We derived an equality between the ratio of the biases that could fit our analysis, but more error analysis is required.
