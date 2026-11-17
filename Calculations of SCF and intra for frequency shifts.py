import numpy as np
from pyscf import gto, scf, fci


#  Define the H-He system


R = 2.3  # bohr

mol = gto.M(
    atom=[
        ['H', (0, 0, 0)],
        ['He', (0, 0, R)]
    ],
    basis='aug-cc-pvdz',
    unit='Bohr',
    charge=0,
    spin=1,
    verbose=0
)


#  Calculate isolated H spin density


H_atom = gto.M(
    atom='H 0 0 0',
    basis='aug-cc-pvdz',
    unit='Bohr',
    spin=1,
    verbose=0
)

hf_H = scf.UHF(H_atom)
hf_H.kernel()

if not hf_H.converged:
    raise RuntimeError("Isolated H SCF failed")

# Evaluate Gaussian orbitals at H nucleus
chi_H = H_atom.eval_gto(
    'GTOval_sph', [[0, 0, 0]]
)[0]

DHa, DHb = hf_H.make_rdm1()

rho0 = np.einsum(
    'i,ij,j->',
    chi_H, DHa - DHb, chi_H
)


#  Unrestricted Hartree-Fock for H-He


hf = scf.UHF(mol)
hf.conv_tol = 1e-11
hf.kernel()

if not hf.converged:
    raise RuntimeError("H-He SCF failed")

Da, Db = hf.make_rdm1()

# Gaussian orbitals at hydrogen nucleus
chi = mol.eval_gto(
    'GTOval_sph', [[0, 0, 0]]
)[0]

rho_SCF = np.einsum(
    'i,ij,j->',
    chi, Da - Db, chi
)

# Calculate SCF fractional shift
f_SCF = rho_SCF / rho0 - 1


# Full configuration interaction


ci = fci.FCI(hf)

E_CI, wavefunction = ci.kernel()

norb = hf.mo_coeff[0].shape[1]

# Correlated spin-resolved density matrices
D_CIa, D_CIb = ci.make_rdm1s(
    wavefunction, norb, mol.nelec
)

# Convert MO density matrices into AO basis
Ca, Cb = hf.mo_coeff

D_CIa = Ca @ D_CIa @ Ca.T
D_CIb = Cb @ D_CIb @ Cb.T

# Correlated spin density at H nucleus
rho_CI = np.einsum(
    'i,ij,j->',
    chi, D_CIa - D_CIb, chi
)

f_CI = rho_CI / rho0 - 1


#  Calculate correlation correction


f_correction = f_CI - f_SCF


# Display results


print("Separation:", R, "bohr")
print("Reference density:", rho0)
print("SCF spin density:", rho_SCF)
print("FCI spin density:", rho_CI)

print("SCF shift x 1e6:", f_SCF * 1e6)
print("FCI correction x 1e6:", f_correction * 1e6)
print("FCI total x 1e6:", f_CI * 1e6)



