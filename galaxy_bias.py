import numpy as np
from matplotlib import pyplot as plt

#Creating arrays for analysis

redshift_bins= np.linspace(0, 0.48275862, 29)

error_bias= np.zeros(len(redshift_bins), dtype=float)
bias= np.zeros(len(redshift_bins), dtype=float)

ratio_refint_squared= np.zeros(len(redshift_bins), dtype=float)
ratio_refint= np.zeros(len(redshift_bins), dtype=float)
error_ratio_refint_squared= np.zeros(len(redshift_bins), dtype=float)
error_ratio_refint= np.zeros(len(redshift_bins), dtype=float)

#Arrays from previous run

interloper_fraction = np.array([6.72583188e-05, 6.90581893e-04, 7.36999606e-04, 1.00200686e-03,
       1.46168431e-03, 1.94149189e-03, 2.19986804e-03, 2.88926581e-03,
       3.17250859e-03, 4.02105280e-03, 4.87054431e-03, 4.99345652e-03,
       5.94525646e-03, 6.50937377e-03, 6.57781622e-03, 6.32441340e-03,
       6.55366006e-03, 6.26283888e-03, 5.84744771e-03, 5.60801757e-03,
       4.93803946e-03, 4.39381546e-03, 3.43798949e-03, 2.79808816e-03,
       2.18092204e-03, 1.62367265e-03, 1.11852479e-03, 7.13080275e-04,
       5.02069141e-04])

omega_auto = np.array([2.05012763, 1.30338729, 1.27975626, 0.91370816, 1.01234443,
       0.92421154, 1.0768    , 1.01248383, 1.14878006, 1.12718913,
       1.10114041, 1.25000176, 1.2817155 , 1.37569732, 1.41812217,
       1.5680806 , 1.71689818, 1.83894649, 2.000722  , 2.28008666,
       2.61843183, 2.67581104, 3.35368152, 3.20192859, 2.93342985,
       2.27165042, 2.98774379, 2.88987801, 9.15514994])

omega_int = np.array([8.49713565, 1.20690262, 1.36481265, 0.83451933, 0.89053876,
       0.72444092, 0.85057375, 0.77146237, 0.8218582 , 0.80590565,
       0.77907609, 0.84735472, 0.84435729, 0.86518735, 0.85708235,
       0.88876844, 0.91826927, 0.89132858, 0.93248075, 0.99488173,
       1.05037828, 1.14741202, 1.27402733, 1.25363404, 1.30743343,
       1.49625615, 1.81769159, 1.56006455, 1.59135223])

ratio_omega = np.array([ 0.00099   ,  0.00307531,  0.00153109, -0.00081839,  0.00139835,
        0.00068268,  0.00260057,  0.00316894,  0.00264345,  0.00249082,
        0.00364291,  0.00342282,  0.00383092,  0.00382291,  0.00584612,
        0.0066684 ,  0.00477584,  0.00424527,  0.00541288,  0.00353843,
        0.00359324,  0.00287338,  0.00289439,  0.00214058,  0.00134525,
        0.00010329,  0.00111482,  0.00355184,  0.00015209])

error_auto = np.array([3.02067661e-02, 9.05972964e-03, 9.44439447e-03, 6.02875659e-03,
       3.32376261e-03, 2.02626986e-03, 1.89477117e-03, 1.48956623e-03,
       1.54168087e-03, 1.32576339e-03, 1.20338205e-03, 1.34436263e-03,
       1.30593357e-03, 1.40593836e-03, 1.71137930e-03, 2.23211386e-03,
       2.84746423e-03, 3.96070519e-03, 5.76314551e-03, 8.61295993e-03,
       1.41727454e-02, 2.39045190e-02, 4.64827984e-02, 8.51048456e-02,
       1.59101780e-01, 2.56671568e-01, 5.12508672e-01, 9.84231817e-01,
       3.68981489e+00])

error_int = np.array([1.83851158, 0.07493574, 0.07631942, 0.04877891, 0.0338357 ,
       0.02427936, 0.0228806 , 0.01698404, 0.01575936, 0.01252172,
       0.01023374, 0.01026369, 0.00867226, 0.00798931, 0.00791532,
       0.00838265, 0.00823064, 0.00855021, 0.00932442, 0.00990506,
       0.0113754 , 0.01332442, 0.01780738, 0.02175643, 0.02849393,
       0.04037424, 0.06303754, 0.09343853, 0.13498057])

error_ratio_omega = np.array([3.72077833e-04, 3.47028163e-04, 3.58733741e-04, 4.16837878e-04,
       2.76736565e-04, 2.37611907e-04, 1.92362246e-04, 1.82155951e-04,
       1.59517450e-04, 1.51112930e-04, 1.47258000e-04, 1.34254108e-04,
       1.28058638e-04, 1.22046331e-04, 1.29710611e-04, 1.31401043e-04,
       1.32324089e-04, 1.43471461e-04, 1.56672260e-04, 1.62884711e-04,
       1.77485837e-04, 2.23809229e-04, 2.38659465e-04, 3.42284843e-04,
       5.26695001e-04, 8.89820987e-04, 9.28347805e-04, 1.78531133e-03,
       6.10262601e-04])

#Bias calculation

for i in range(len(redshift_bins)):

    bias[i]= interloper_fraction[i]/ratio_omega[i]

    error_bias[i]= np.absolute(interloper_fraction[i]*(error_ratio_omega[i]/np.power(ratio_omega[i],2)))


    ratio_refint_squared[i] = omega_auto[i]/omega_int[i]
    ratio_refint[i] = np.sqrt(omega_auto[i]/omega_int[i])

    error_ratio_refint_squared[i]= np.absolute(ratio_refint_squared[i]) * np.sqrt(np.power(error_int[i]/omega_int[i],2)+np.power(error_auto[i]/omega_auto[i],2))

    error_ratio_refint[i]= np.absolute(0.5 * (error_ratio_refint_squared[i]/ratio_refint_squared[i]) * ratio_refint[i])

# Plotting
plt.figure(figsize=(12, 8))

# Line plot for bias and ratio (with error bars)
#plt.errorbar(redshift_bins, bias, yerr=error_bias, fmt='-', color='blue', label=r'Bias')
plt.errorbar(redshift_bins, ratio_refint, yerr=error_ratio_refint, fmt='-', color='green', label=r'Bias ratio of interlopers and reference auto correlations')

# Y-axis limits
plt.ylim(0, 2.5)

# Title and axis labels with LaTeX
plt.title(r'Bias and Correlation Analysis', fontsize=20, fontweight='bold')
plt.xlabel(r'True Redshift Bins', fontsize=16)
plt.ylabel(r'$\sqrt{\frac{\omega_{\text{auto}}}{\omega_{\text{int}}}}$', fontsize=20)

# Adding a legend
plt.legend(fontsize=14)

# Show the plot
plt.show()


