import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci

# H-He separations in bohrs from Table 5
R_values = np.array([2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5,
                     5.5, 6, 6.5, 6.75, 7, 7.5, 8, 9, 10, 11, 12, 15])

#Total values in table 5
paper_total = np.array([
    -51451.52, 3804.23, 27096.20, 33415.79, 31647.77,
    21066.20, 11538.65, 5620.32, 2483.23, 982.19, 325.08,
    63.81, 5.75, -26.25, -48.29, -46.50, -29.22, -15.87,
    -8.97, -5.08, -1.25
])

basis = "aug-cc-pvdz"


def spin_density(mol, dm_alpha, dm_beta):
    phi = mol.eval_gto("GTOval", mol.atom_coord(0).reshape(1, 3))[0]
    return float(np.einsum("i,ij,j->", phi, dm_alpha - dm_beta, phi).real)


# Calculate the isolated hydrogen reference once.
h = gto.M(atom="H 0 0 0", unit="Bohr", basis=basis, spin=1, verbose=0)
hf_h = scf.UHF(h).run()
if not hf_h.converged:
    raise RuntimeError("Isolated H calculation did not converge")
rho0 = spin_density(h, *hf_h.make_rdm1())

calculated_total = np.full(len(R_values), np.nan)

for i, R in enumerate(R_values):
    print(f"Calculating R = {R:g} bohr ...", flush=True)
    try:
        mol = gto.M(atom=[["H", (0, 0, 0)], ["He", (0, 0, float(R))]],
                    basis=basis, unit="Bohr", spin=1, verbose=0)

        uhf = scf.UHF(mol).run()
        rohf = scf.ROHF(mol).run()
        if not (uhf.converged and rohf.converged):
            raise RuntimeError("SCF did not converge")

        solver = fci.FCI(mol, rohf.mo_coeff)
        _, ci_vec = solver.kernel()
        if not np.all(solver.converged):
            raise RuntimeError("FCI did not converge")

        n_orb = rohf.mo_coeff.shape[1]
        dm_a_mo, dm_b_mo = solver.make_rdm1s(ci_vec, n_orb, mol.nelec)
        C = rohf.mo_coeff
        dm_a = C @ dm_a_mo @ C.T.conj()
        dm_b = C @ dm_b_mo @ C.T.conj()
        rho = spin_density(mol, dm_a, dm_b)

        #turning it into the same units as in table 5 where they have been
        #multiplied by 10^6
        calculated_total[i] = 1e6 * (rho / rho0 - 1)
        print(f"  Calculated: {calculated_total[i]:.2f} | Paper: {paper_total[i]:.2f}")
    except Exception as error:
        print(f"  Failed at R={R:g}: {error}")

#Plotting the graphs against eachother
plt.figure(figsize=(9, 5))
plt.plot(R_values, paper_total, "ko-", label="Meyer & Frommhold (Table 5)")
plt.plot(R_values, calculated_total, "rs--", label="My PySCF calculations")
plt.axhline(0, color="grey", linewidth=0.8)
plt.yscale("symlog", linthresh=10)  # makes positive and negative small shifts visible
plt.xlabel("H-He separation R (bohr)")
plt.ylabel(r"Relative frequency shift $10^6 \Delta\nu/\nu_0$")
plt.title("Hydrogen-helium hyperfine shift")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
