"""
These are calculations using equation 9 referred to in the Meyer Frommhold
paper as the semi classical approach and as quoted'At temperatures T > 40 K, 
the semi-classical form of the pair distribution form may be used,'
so this can only be used to compare to the data in Table 7 for temperatures
greater than 40K
"""

import numpy as np
from scipy.constants import k, hbar, physical_constants
from scipy.integrate import simpson
from scipy.special import gammainc, factorial

#Setting values
T = 300.0                # temperature (kelvin); Eq. (9) suited to T > ~40 K
n_He_cm3 = 1.0e16        # helium number density (atoms/cm^3)
nu0 = 1.420405726e9     # unperturbed H hyperfine frequency (Hz)
R_min = 1.0              # numerical lower bound (bohr)
R_max = 200.0            # numerical upper bound (bohr)
N = 40001                # number of grid points

#Physical constants in Si units
Eh = physical_constants['Hartree energy'][0]      # J
bohr_m = physical_constants['Bohr radius'][0]     # m
amu = physical_constants['atomic mass constant'][0] # kg
m_H = 1.00782503223 * amu
m_He = 4.00260325413 * amu
mu = (m_H * m_He) / (m_H + m_He)  # reduced mass (kg)

#Coefficients as from the Meyer Frommhold paper
# Potential V(R), Table 3
A1, A2, A3 = 1.8399, 0.9450, 2.2152
A4, A5, A6 = 2.2227, 0.9779, 1.8883

# Relative hyperfine shift f(R) = Delta a(R)/a_inf, Table 6
C1, C2, C3 = -14.530, 1.15320, 3.33048
C4, C5, C6 = 2.22110, 0.98313, 1.53633


def higher_coefficients(v6, v8, v10):
    """Generating coefficients for K and C using equation 4"""
    result = {6: v6, 8: v8, 10: v10}
    for p in range(12, 20, 2):
        result[p] = (result[p - 2] / result[p - 4])**3 * result[p - 6]
    return result


C = higher_coefficients(2.823, 41.83, 871.3)   # potential coefficients
K = higher_coefficients(13.26, 269.0, 7221.0) # hyperfine coefficients from table 6


def potential_and_derivatives(R):
    """Returning potential and derivatives in hartree energy units using equations 1, 2 and 3."""
    R = np.asarray(R, dtype=float)
    if np.any(R <= 0):
        raise ValueError('All R values must be positive.')

    # Repulsive part: Equation (1), analytically differentiated
    """Differential workings in my work paper book"""
    a = A1 + A2 * R**A3
    da = A2 * A3 * R**(A3 - 1)
    d2a = A2 * A3 * (A3 - 1) * R**(A3 - 2)
    e = np.exp(-A4 * R)

    V = a * e
    Vp = (da - A4*a) * e
    Vpp = (d2a - 2*A4*da + A4**2*a) * e

    # Attractive part, Equations (2) (3), analytically differentiated
    x = A6 * R
    for p, Cp in C.items():
        D = gammainc(p + 1, x)                    # D_p(x)
        Dx = np.exp(-x) * x**p / factorial(p)    # dD_p/dx
        Dxx = Dx * (p/x - 1)                    # d2D_p/dx2
        q = -A5 * Cp
        V += q * D / R**p
        Vp += q * (x*Dx - p*D) / R**(p + 1)
        Vpp += q * (x*x*Dxx - 2*p*x*Dx + p*(p + 1)*D) / R**(p + 2)

    return V, Vp, Vpp


def relative_hyperfine_shift(R):
    """calculating the freuency using equation 6"""
    R = np.asarray(R, dtype=float)
    f = (C1 + C2 * R**C3) * np.exp(-C4 * R)
    for p, Kp in K.items():
        D = gammainc(p + 1, C6 * R)         #"""D is calculated from the incomplete gamma damping function"""
        f -= C5 * Kp * D / R**p
    return f


def pair_distribution(R, temperature):
    """Calculating g from equation 9 using the correct units"""
    V, Vp, Vpp = potential_and_derivatives(R)
    R_m = R * bohr_m

    # Convert from atomic units to SI so that SI hbar can be used.
    V_J = V * Eh
    Vp_SI = Vp * Eh / bohr_m
    Vpp_SI = Vpp * Eh / bohr_m**2
    kT = k * temperature

    # Full Eq. (9) through order hbar**2 (not the classical-only result)
    bracket = Vpp_SI + 2*Vp_SI/R_m - Vp_SI**2/(2*kT)
    correction = 1 - hbar**2 * bracket / (12*mu*kT**2)
    g = np.exp(-V_J / kT) * correction
    return g


#Integrating equation 7 to find the temperature dependent frequency shifts across a range of R
R = np.linspace(R_min, R_max, N)
R_m = R * bohr_m
f = relative_hyperfine_shift(R)
g = pair_distribution(R, T)

integral = simpson(f * g * R_m**2, x=R_m)
n_He_m3 = n_He_cm3 * 1e6  # atoms/cm^3 -> atoms/m^3

# Pressure-induced frequency shift from the full Eq. (9), in Hz
delta_nu = 4 * np.pi * n_He_m3 * nu0 * integral



print(f'Frequency shift using full Eq. (9): {delta_nu:.8f} Hz')
print(f'New hyperfine transition frequency: {nu0 + delta_nu:.8f} Hz')
print(f"Table 7 units value: {delta_nu*100:.4f}")

#------------------------------------------------------------------------------
"""
Calculating just classicaly not using equation 9
"""
#------------------------------------------------------------------------------


# Calculate interaction potential
V, _, _ = potential_and_derivatives(R)

# Convert potential from Hartrees to Joules
V_J = V * Eh

# Classical pair distribution
g_classical = np.exp(-V_J / (k * T))

# Integrate Equation (7)
integral_classical = simpson(
    f * g_classical * R_m**2, x=R_m
)

# Calculate frequency shift in Hz
delta_nu_classical = (
    4 * np.pi * n_He_m3 * nu0 * integral_classical
)

# Convert to Table 7 units
coefficient = delta_nu_classical / n_He_cm3 * 1e18


print()
print(f"Temperature: {T} K")
print(f"Classical frequency shift: {delta_nu_classical:.8f} Hz")
print(f"New transition frequency: {nu0 + delta_nu_classical:.8f} Hz")
print(f"Table 7 units value: {coefficient:.4f}")

