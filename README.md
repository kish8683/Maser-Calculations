# Maser-Calculations
Python files used to calculate different physical values of hydrogen/tritium masers.

Meyer_frommhold_classical is the file used for calculating both the classical and semiclassical values for the thermally averaged hyperfine pressure shifts. The semi-classical results will not be correct for temperatures under 40K

Compareing_table5_pySCF is the file used to compare the data given from the pySCF function to the values of the frequency shift shown in the paper.

Quantum_shifts.py is the file used for the calculation of the quantum thermally averaged hyperfine pressure shifts. It uses equations 1,2,3,4,6,7 and 8 from the Meyer Frommhold paper to calculate the shifts. The numerov mathematical recurrence relation is used to solve the wavefunction numverically. Then the radial integral is solved using simpson integration followed by the sum from equation 8 and lastly the Gauss-laguerre to do the last integration that involves an e^-x function.
