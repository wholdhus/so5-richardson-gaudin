import sys
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_eqs import np, solve_rgEqs, calculate_energies

RESULT_FP = '/home/wholdhus/so5_results/'

if len(sys.argv) < 4:
    print('Usage: python rg_hpc.py [L] [N] [G]')
    print('L: number of levels')
    print('N: total number of electrons (half up, half down)')
    print('G: desired final coupling')

L = int(sys.argv[1])
N = int(sys.argv[2])
gf = float(sys.argv[3])
Ne = N//2
Nw = N//2


dg = 0.0005*8/L
g0 = dg/N
imk = g0
imv = 0.1*g0

ks = (1.0*np.arange(L) + 1.0)/L

print('Parameters:')
print(L)
print(N)
print(gf)
print(ks)

VERBOSE=True
FORCE_GS=True
TOL=10**-10
TOL2=10**-7 # there are plenty of spurious minima around 10**-5
MAXIT=0 # let's use the default value
FACTOR=100
JOBS = 64

dims = (L, Ne, Nw)
es, ws, vars_df, varss = solve_rgEqs(dims, gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                    imscale_v=imv)
print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'solutions_{}_{}_{}.csv'.format(L, N, gf))

energies = calculate_energies(varss, vars_df['g'], ks, Ne)
plt.scatter(vars_df['g'], energies)
plt.savefig(RESULT_FP + 'energies_{}_{}_{}.png'.format(L, N, gf))
