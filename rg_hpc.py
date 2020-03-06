import sys
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_eqs import np, solve_rgEqs, calculate_energies, G_to_g

RESULT_FP = '/home/wholdhus/so5_results/'

if len(sys.argv) < 4:
    print('Usage: python rg_hpc.py [L] [N] [G]')
    print('L: number of levels')
    print('N: total number of electrons (half up, half down)')
    print('G: desired final coupling')

L = int(sys.argv[1])
N = int(sys.argv[2])
Gf = float(sys.argv[3])
Ne = N//2
Nw = N//2


dg = 0.0005*8/L
g0 = dg/N
imk = g0
imv = 0.1*g0

ks = (1.0*np.arange(L) + 1.0)/L

gf = G_to_g(Gf, ks)

print('')
print('Parameters:')
print('Length')
print(L)
print('Fermions')
print(N)
print('Final coupling (physical)')
print(Gf)
print('Final coupling (numerical)')
print(gf)
print('Spectrum')
print(ks)
print('Imaginary part of guesses')
print(imv)
print('')

dims = (L, Ne, Nw)

es, ws, vars_df = solve_rgEqs(dims, gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                    imscale_v=imv)
print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'solutions_full_{}_{}_{}.csv'.format(L, N, gf))

energies = vars_df['energy']

plt.scatter(vars_df['G'], energies)
plt.savefig(RESULT_FP + 'energies_full_{}_{}_{}.png'.format(L, N, gf))
