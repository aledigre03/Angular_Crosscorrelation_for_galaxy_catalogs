#!/usr/bin/env python

from math import sqrt, pi, sin, sinh

# Constants
H0 = 69.6  # Hubble constant in km/s/Mpc
WM = 0.315  # Omega matter
WV = 1.0 - WM  # Omega vacuum (assuming a flat universe)
c = 299792.458  # Speed of light in km/s
z1 = 1.0
z2 = 1.1


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
V_Gpc1 = (( 4.0 * pi * ((0.001 * c / H0) ** 3) * VCM1 )/8) * h**3 # Comoving volume in Gpc^3/h^3

VCM2 = ratio2 * DCMR2 * DCMR2 * DCMR2 / 3.0
V_Gpc2 = (( 4.0 * pi * ((0.001 * c / H0) ** 3) * VCM2 )/8) * h**3 # Comoving volume in Gpc^3/h^3

V_Gpc= (V_Gpc2-V_Gpc1)
# Output the result
print(f'The comoving volume within redshift z={z1} and z= {z2} for an octant of the sky is {V_Gpc:.3f} Gpc^3/h^3.')
