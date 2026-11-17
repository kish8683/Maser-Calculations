import numpy as np
from scipy.constants import k as k_B, physical_constants
from scipy.special import gammainc, roots_laguerre, spherical_jn, spherical_yn
import matplotlib.pyplot as plt

#INPUTS
n_He_cm3 = 1.0e16          # Illustrative He number density, atoms per cm^3
TEMPERATURES = [0.5, 0.7, 1.0, 1.15, 2.0, 3.0, 4.0, 5.0,
        10.0, 12.5, 16.0, 20.0, 30.0, 40.0, 50.0,
        100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]

# Accuracy controls. For extra accuracy, increase N_ENERGY and
# decrease DR, then make sure the reported result has converged.
N_ENERGY = 64               # Gauss Laguerre number of steps in the sum
R_MIN = 1.5                 # minimum distance, this is inside the core
R_MAX = 100.0               # maximum distance, won't change at much larger
DR = 0.025                 # step sizes
ACTIVE_RADIUS = 50.0       # distance for partial wave checking
ELL_MARGIN = 12
ENERGY_WINDOW = 18.0       

#Physical constants
Eh = physical_constants["Hartree energy"][0]              # J
bohr_cm = physical_constants["Bohr radius"][0] * 100     # cm
me = physical_constants["electron mass"][0]              # kg
amu = physical_constants["atomic mass constant"][0]      # kg
nu0 = 1.420405726e9                                      # Hz

m_H = 1.00782503223 * amu
m_He = 4.00260325413 * amu

#Reduced mass in electron mass units
mu = (m_H * m_He / (m_H + m_He)) / me


#H--He potential calculated using equations 1 to 4

C = {6: 2.823, 8: 41.83, 10: 871.3}
for p in range(12, 20, 2):
    C[p] = (C[p - 2] / C[p - 4])**3 * C[p - 6]


def interaction_potential(R):
    """Calculating the total potentials in hartrees"""
    R = np.asarray(R, dtype=float)
    V = (1.8399 + 0.9450 * R**2.2152) * np.exp(-2.2227 * R)
    for p, Cp in C.items():
        # Eq. (3): lower regularised incomplete gamma damping
        D = gammainc(p + 1, 1.8883 * R)
        V -= 0.9779 * Cp * D / R**p
    return V


#hyperfine frequency shifts calculated using equation 6 and constants in table 6

K = {6: 13.26, 8: 269.0, 10: 7221.0}
for p in range(12, 20, 2):
    K[p] = (K[p - 2] / K[p - 4])**3 * K[p - 6]


def relative_hyperfine_shift(R):
    """hyperfine frequency shifts"""
    R = np.asarray(R, dtype=float)
    f = (-14.530 + 1.15320 * R**3.33048) * np.exp(-2.22110 * R)
    for p, Kp in K.items():
        D = gammainc(p + 1, 1.53633 * R)
        f -= 0.98313 * Kp * D / R**p
    return f



#calculating it all

def quantum_frequency_shift(temperature, density_cm3):
    if temperature <= 0 or density_cm3 < 0:
        raise ValueError("Temperature must be positive and density 0 or above.")

    #All energies are in Hartrees from this point onward.
    kT = k_B * temperature / Eh

    # Gauss-Laguerre quadrature directly integrates the Boltzmann
    x_nodes, weights = roots_laguerre(N_ENERGY) #picking the correct number of integration 
    #points and weights that work for the e^-x 
    collision_energies = kT * x_nodes      #turning the integration points into energies
    q = np.sqrt(2.0 * mu * collision_energies)  #calculating wave number

    #calculating the number of values of l we will sum over in equation 8
    ell_max = max(
        15,
        int(np.ceil(np.sqrt(2 * mu * kT * ENERGY_WINDOW)
                    * ACTIVE_RADIUS + ELL_MARGIN))
    )
    ell = np.arange(ell_max + 1) #making an array from 0 to ell_max
    active = ell[:, None] <= q[None, :] * ACTIVE_RADIUS + ELL_MARGIN


    n_steps = int(np.ceil((R_MAX - R_MIN) / DR))
    if n_steps % 2:     #making sure the number of steps is even
        n_steps += 1
    R = np.linspace(R_MIN, R_MAX, n_steps + 1)      #creating the grid of R values
    h = R[1] - R[0]     #finding the difference between adjacent points
    V = interaction_potential(R)        #calling back the functions for potentials that use
    f = relative_hyperfine_shift(R) #euations 1-4 and 6

    # Schrodinger radial equation u'' = F(R) u:
    # F = 2*mu*(V-E) + ell*(ell+1)/R^2.
    L = ell[:, None] * (ell[:, None] + 1)   #doing L = l(l+1)
    twice_mu_E = 2 * mu * collision_energies[None, :]   #calculating the 2muE part of the wavefunction

    def F_at(i):
        return 2 * mu * V[i] - twice_mu_E + L / R[i]**2 #Writing the equation for function F

    #boundary condition that wavefunction is 0 within the core
    #sets the second step at 10^-19
    u_previous = np.zeros((ell_max + 1, N_ENERGY))
    u_current = np.where(active, 1.0e-19, 0.0)

    #making the integrand part of the radial integral
    radial_integrals = np.zeros_like(u_current)
    radial_integrals += 4.0 * h / 3.0 * f[1] * u_current**2

    F_previous = F_at(0)
    F_current = F_at(1)

    #doing the numerov recurrence relation for every value of ell and energy 
    for i in range(1, n_steps):
        F_next = F_at(i + 1)

        u_next = (
            2 * (1 + 5 * h**2 * F_current / 12) * u_current
            - (1 - h**2 * F_previous / 12) * u_previous
        ) / (1 - h**2 * F_next / 12)

        # Simpson integration, accumulated while propagating.
        simpson_weight = 1 if i + 1 == n_steps else (4 if (i + 1) % 2 else 2)   #setting the weights for the simpson integral
        radial_integrals += (
            h / 3 * simpson_weight * f[i + 1] * u_next**2
        )   #adding the calculated value for each ell and energy to the total for the radial integral

        #Preventing some values from getting too big so rescales the amplitudes every 25 times
 
        if i % 25 == 0:
            huge = np.abs(u_next) > 1.0e65
            if np.any(huge):
                u_current[huge] *= 1.0e-60
                u_next[huge] *= 1.0e-60
                radial_integrals[huge] *= 1.0e-120

        u_previous, u_current = u_current, u_next#moving along to the next point 
        F_previous, F_current = F_current, F_next



    weighted_strengths = np.zeros(N_ENERGY)

    for j in range(N_ENERGY):   #normalising the wavefunctions
        ls = ell[active[:, j]]
        z0 = q[j] * R[-1]
        z1 = q[j] * R[-2]

        s0 = z0 * spherical_jn(ls, z0)  #matching the wavefunctions to the asymptotic solutions
        c0 = -z0 * spherical_yn(ls, z0)
        s1 = z1 * spherical_jn(ls, z1)
        c1 = -z1 * spherical_yn(ls, z1)

        determinant = s0 * c1 - s1 * c0
        a = (u_current[ls, j] * c1 - u_previous[ls, j] * c0) / determinant
        b = (s0 * u_previous[ls, j] - s1 * u_current[ls, j]) / determinant
        amplitude_squared = a**2 + b**2

        if np.any(~np.isfinite(amplitude_squared)) or np.any(amplitude_squared <= 0):
            raise ArithmeticError("failure in the normalisation part.")

        # Delta(E-E') normalization in atomic units:
        # asymptotic amplitude squared = 2*mu/(pi*q).
        energy_norm_sq = 2 * mu / (np.pi * q[j])    #normalising the wavefunctions
        radial_norm = radial_integrals[ls, j] * energy_norm_sq / amplitude_squared
        weighted_strengths[j] = np.sum((2 * ls + 1) * radial_norm)  #summing the radial integral of wavefunctions
                                                                    #times by (2l+1) from equation 8

    #in equation 8 lambda cumbed is used
    lambda_cubed = (2 * np.pi / (mu * kT))**1.5

    #now doing the last integral over energy of (e^-E/kT) * the sum over all values of l of the radial integral
    energy_integral = kT * np.dot(weights, weighted_strengths)
    shift_per_density_au = nu0 * lambda_cubed * energy_integral

    # Convert density from cm^-3 to bohr^-3.
    density_au = density_cm3 * bohr_cm**3
    delta_nu_hz = density_au * shift_per_density_au

    # Table 7 coefficient, in 10^-18 Hz cm^3.
    table7_coefficient = shift_per_density_au * bohr_cm**3 * 1.0e18
    return delta_nu_hz, table7_coefficient, ell_max



#----------------------------RESULTS-------------------------------------------
#Also plotting a graph of the results against eachother

if __name__ == '__main__':
    # These are the quantum entries in the Meyer and Frommhold paper in Table 7,

    paper_table7 = {
    0.5: -8.06, 0.7: -8.20, 1.0: -8.49,
    1.15: -8.53, 2.0: -8.41, 3.0: -7.92,
    4.0: -7.14, 5.0: -6.16, 10.0: -1.36,
    12.5: 1.13, 16.0: 4.59, 20.0: 8.42,
    30.0: 16.75, 40.0: 25.15, 50.0: 33.82,
    100.0: 71.19, 150.0: 102.8, 200.0: 130.9,
    250.0: 156.4, 300.0: 180.1, 350.0: 202.1,
    400.0: 222.0
}

    results = []
    print('H - 4He quantum hyperfine pressure shift, Meyer & Frommhold')
    print('All pressure-shift coefficients: 10^-18 Hz cm^3')
    print('T(K)    Calculated       Paper Table 7')
    for temp in TEMPERATURES:
        shift_hz, coefficient, ell_max = quantum_frequency_shift(temp, n_He_cm3)
        published = paper_table7.get(temp, float('nan'))
        print(f'{temp:5.1f}    {coefficient:11.4f}    {published:11.4f}')
        results.append((temp, coefficient, published, shift_hz, ell_max))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    temps, calc, reference, _, _ = np.array(results).T
    ax.plot(temps, calc, 'o-', label='Numerov + quantum thermal average')
    ax.plot(temps, reference, 's--', label='Meyer–Frommhold Table 7')
    ax.axhline(0, color='gray', linewidth=0.7)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel(r'$\Delta\nu/n_{He}$ ($10^{-18}$ Hz cm$^3$)')
    ax.set_title('H–He quantum hyperfine pressure shift')
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    plt.show()  